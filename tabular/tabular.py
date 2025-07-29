import os
import logging
import hail as hl
import input.dataset as ds
from tabular.hail_utils import init_hail, get_temp_path, clean, TabularProcessor


def parse_ldr_col(ldr_col):
    """
    Parsing string for LDR indices

    Parameters:
    ------------
    ldr_col: a string of one-based LDR column indices

    Returns:
    ---------
    res: a tuple of min and max (not included) zero-based indices

    """
    ldr_col = ldr_col.split(",")
    res = list()

    for col in ldr_col:
        if ":" in col:
            start, end = [int(x) for x in col.split(":")]
            if start > end:
                raise ValueError(f"{col} is invalid")
            res += list(range(start - 1, end))
        else:
            res.append(int(col) - 1)

    res = sorted(list(set(res)))
    if res[-1] - res[0] + 1 != len(res):
        raise ValueError(
            "it is very rare that columns in --ldr-col are not consective for LDR GWAS"
        )
    if res[0] < 0:
        raise ValueError("the min index less than 1")
    res = (res[0], res[-1] + 1)

    return res


def pandas_to_table(df, dir):
    """
    Converting a pd.DataFrame to hail.Table

    Parameters:
    ------------
    df: a pd.DataFrame to convert, it must have a single index 'IID'

    Returns:
    ---------
    table: a hail.Table

    """
    if not df.index.name == "IID":
        raise ValueError("the DataFrame must have a single index IID")
    df.to_csv(f"{dir}.txt", sep="\t", na_rep="NA")

    table = hl.import_table(
        f"{dir}.txt", key="IID", impute=True, types={"IID": hl.tstr}, missing="NA"
    )

    return table


def check_input(args, log):
    # required arguments
    if args.ldrs is None:
        raise ValueError("--ldrs is required")
    if args.covar is None:
        raise ValueError("--covar is required")
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")
    if args.tabular_mt is None:
        raise ValueError(
            "--tabular-mt is required. If you have tabular data, convert it into a mt by --make-mt"
        )

    if args.ldr_col is not None:
        args.ldr_col = parse_ldr_col(args.ldr_col)
        if args.n_ldrs is not None:
            log.info(
                "WARNING: ignoring --n-ldrs as --ldr-col has been provided."
            )
    elif args.n_ldrs is not None:
        args.ldr_col = (0, args.n_ldrs)
    args.n_ldrs = None


class DoAnalysis:
    """
    Conducting association analysis for LDRs

    """

    def __init__(self, mt_processor, ldrs, covar, temp_path, rand_v=1):
        """
        Parameters:
        ------------
        mt_processor: 
        ldrs: a pd.DataFrame of LDRs with a single index 'IID'
        covar: a pd.DataFrame of covariates with a single index 'IID'
        temp_path: a temporary path for saving interim data
        rand_v (n, 1): a np.array of random standard normal variable for wild bootstrap

        """
        self.mt_processor = mt_processor
        self.ldrs = ldrs
        self.covar = covar
        self.n_ldrs = self.ldrs.shape[1]
        self.n_covar = self.covar.shape[1]
        self.temp_path = temp_path
        self.logger = logging.getLogger(__name__)
        self.n_variables = self.mt_processor.mt.count_rows()

        covar_table = pandas_to_table(self.covar, f"{temp_path}_covar")
        self.mt_processor.annotate_cols(covar_table, "covar")

        self.logger.info(
            (f"Doing association analysis for {self.n_variables} variables "
             f"and {self.n_ldrs} LDRs ...")
        )
        ldrs_table = pandas_to_table(self.ldrs * rand_v, f"{temp_path}_ldr")
        self.mt_processor.annotate_cols(ldrs_table, "ldrs")
        self.mt_processor.mt = self.mt_processor.mt.annotate_rows(
            n_called = hl.agg.count_where(hl.is_defined(self.mt_processor.mt.value))
        )
        self.results = self.do_analysis(self.mt_processor.mt)

    def do_analysis(self, mt):
        """
        Conducting association analysis for all LDRs

        Parameters:
        ------------
        mt: a hail.MatrixTable with LDRs and covariates annotated

        Returns:
        ---------
        results: results in hail.Table

        """
        pheno_list = [mt.ldrs[i] for i in range(self.n_ldrs)]
        covar_list = [mt.covar[i] for i in range(self.n_covar)]

        results = hl.linear_regression_rows(
            y=pheno_list,
            x=mt.value,
            covariates=covar_list,
            pass_through=[mt.n_called, mt.variable],
        )

        results = results.key_by()
        results = results.drop(*["y_transpose_x", "sum_x"])
        results = results.select(
            "variable",
            "n_called",
            "beta",
            "standard_error",
            "t_stat",
            "p_value",
        )
        # TODO: 12:26am, get n_called when generating mt.
        # need to add an option in sumstats and voxel-results

        results = self._post_process(results)

        return results

    def _post_process(self, results):
        """
        Removing SNPs with any missing or infinity values.
        This step is originally done in sumstats.py.
        However, pandas is not convenient to handle nested arrays.

        """
        results = results.filter(
            ~(
                hl.any(lambda x: hl.is_missing(x) | hl.is_infinite(x), results.beta)
                | hl.any(
                    lambda x: hl.is_missing(x) | hl.is_infinite(x), results.standard_error
                )
                | hl.any(lambda x: hl.is_missing(x) | hl.is_infinite(x), results.t_stat)
                | hl.any(lambda x: hl.is_missing(x) | hl.is_infinite(x), results.p_value)
            )
        )

        return results

    def save(self, out_path):
        """
        Saving results as a parquet file

        """
        self.results = self.results.to_spark()
        self.results.write.mode("overwrite").parquet(f"{out_path}.parquet")


def run(args, log):
    # check input and configure hail
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.out)

        # read LDRs and covariates
        log.info(f"Read LDRs from {args.ldrs}")
        ldrs = ds.Dataset(args.ldrs)
        log.info(f"{ldrs.data.shape[1]} LDRs and {ldrs.data.shape[0]} subjects.")
        if args.ldr_col is not None:
            if ldrs.data.shape[1] < args.ldr_col[1]:
                raise ValueError(f"--ldr-col or --n-ldrs out of index")
            else:
                log.info(f"Keeping LDR{args.ldr_col[0]+1} to LDR{args.ldr_col[1]}.")
            ldrs.data = ldrs.data.iloc[:, args.ldr_col[0] : args.ldr_col[1]]

        log.info(f"Read covariates from {args.covar}")
        covar = ds.Covar(args.covar, args.cat_covar_list)
        common_ids = ds.get_common_idxs(ldrs.data.index, covar.data.index, args.keep)
        common_ids = ds.remove_idxs(common_ids, args.remove, single_id=True)

        # read tabular data
        mt_processor = TabularProcessor(args.tabular_mt)
        log.info(f"Processing tabular data ...")
        mt_processor.keep_remove_idvs(common_ids)

        # extract common subjects and align data
        mt_ids = mt_processor.subject_id()
        ldrs.to_single_index()
        covar.to_single_index()
        ldrs.keep_and_remove(mt_ids)
        covar.keep_and_remove(mt_ids)
        covar.cat_covar_intercept()

        log.info(f"{len(mt_ids)} common subjects in the data.")
        log.info(
            f"{covar.data.shape[1]} fixed effects in the covariates (including the intercept)."
        )

        # assoc analysis
        temp_path = get_temp_path(args.out)
        mt_processor.cache()
        results = DoAnalysis(mt_processor, ldrs.data, covar.data, temp_path)

        # save results
        results.save(args.out)
        log.info(f"\nSaved association results to {args.out}.parquet")
    finally:
        if "temp_path" in locals():
            if os.path.exists(f"{temp_path}_covar.txt"):
                os.remove(f"{temp_path}_covar.txt")
                log.info(f"Removed temporary covariate data at {temp_path}_covar.txt")
            if os.path.exists(f"{temp_path}_ldr.txt"):
                os.remove(f"{temp_path}_ldr.txt")
                log.info(f"Removed temporary LDR data at {temp_path}_ldr.txt")

        clean(args.out)
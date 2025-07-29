import sys
import os
import re
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from tabular import utils


class Dataset:
    def __init__(self, dir, all_num_cols=False):
        """
        Read a dataset and do preprocessing

        Parameters:
        ------------
        dir: diretory to the dataset
        all_num_cols: if all columns are numbers (except for FID and IID)

        """
        self.logger = logging.getLogger(__name__)
        openfunc, compression = utils.check_compression(dir)
        cols = self._check_header(openfunc, compression, dir)

        if not all_num_cols:
            dtype_dict = {"FID": str, "IID": str}
        else:
            dtype_dict = {col: "float32" for col in cols}
            dtype_dict["FID"] = str
            dtype_dict["IID"] = str

        self.data = pd.read_csv(
            dir,
            sep="\s+",
            compression=compression,
            na_values=[-9, "NONE", "."],
            dtype=dtype_dict,
        )

        n_sub = len(self.data)
        self.data.drop_duplicates(subset=["FID", "IID"], inplace=True, keep=False)
        self.logger.info(f"Removed {n_sub - len(self.data)} duplicated subjects.")
        self._remove_na_inf()

        self.data = self.data.set_index(["FID", "IID"])
        self.data = self.data.sort_index()

        if all_num_cols:
            n_sub = len(self.data)
            self.data = self.data[np.std(self.data, axis=1) != 0]
            self.logger.info(
                f"Removed {n_sub - len(self.data)} subjects with zero variance."
            )

    def _check_header(self, openfunc, compression, dir):
        """
        The dataset must have a header: FID, IID, ...

        Parameters:
        ------------
        openfunc: function to read the first line
        compression: how the data is compressed
        dir: diretory to the dataset

        """
        with openfunc(dir, "r") as file:
            header = file.readline().split()
        if len(header) == 1:
            raise ValueError(
                "only one column detected, check your input file and delimiter"
            )
        if compression is not None:
            header = [str(header[i], "UTF-8") for i in range(len(header))]
        if header[0] != "FID" or header[1] != "IID":
            raise ValueError("the first two column names must be FID and IID")
        if len(header) != len(set(header)):
            raise ValueError("duplicated column names are not allowed")

        return header[:2]

    def _remove_na_inf(self):
        """
        Removing rows with any missing/infinite values

        """
        bad_idxs = (self.data.isin([np.inf, -np.inf, np.nan])).any(axis=1)
        self.data = self.data.loc[~bad_idxs]
        self.logger.info(
            f"Removed {sum(bad_idxs)} row(s) with missing or infinite values."
        )

    def keep_and_remove(self, keep_idx=None, remove_idx=None, merge=False):
        """
        Extracting and removing rows using indices (not boolean)
        the resulting dataset should have the same order as the keep_idx

        Parameters:
        ------------
        keep_idx: indices to keep
        remove_idx: indices to remove
        merge: if getting commmon indices before keeping, the data order will be different

        """
        if keep_idx is not None:
            if merge:
                common_idxs = get_common_idxs(self.data.index, keep_idx)
            else:
                common_idxs = keep_idx
            self.data = self.data.loc[common_idxs]
        if remove_idx is not None:
            self.data = self.data[~self.data.index.isin(remove_idx)]
        if len(self.data) == 0:
            raise ValueError("no data left")

    def to_single_index(self):
        """
        Using only the IID as index for compatible with hail

        """
        self.data = self.data.reset_index(level=0, drop=True)
        # self.data.reset_index(inplace=True)

    def get_ids(self):
        return self.data.index


class Covar(Dataset):
    def __init__(self, dir, cat_covar_list=None):
        """
        Parameters:
        ------------
        dir: diretory to the dataset
        cat_covar_list: a string of categorical covariates separated by comma

        """
        super().__init__(dir)
        self.cat_covar_list = cat_covar_list

    def cat_covar_intercept(self):
        """
        Converting categorical covariates to dummy variables,
        and adding the intercept. This step must be done after
        merging datasets, otherwise some dummy variables might
        cause singularity.

        """
        if self.cat_covar_list is not None:
            catlist = self.cat_covar_list.split(",")
            self._check_validcatlist(catlist)
            self.logger.info(
                f"{len(catlist)} categorical variables provided by --cat-covar-list."
            )
            self.data = self._dummy_covar(catlist)
        if (
            not self.data.shape[1]
            == self.data.select_dtypes(include=np.number).shape[1]
        ):
            raise ValueError(
                (
                    "did you forget some categorical variables? Probably you left NAs as blank. "
                    "Fill NAs as `NA`, `-9`, `NONE`, or `.`"
                )
            )
        self._add_intercept()
        if self._check_singularity():
            raise ValueError("the covarite matrix is singular")

    def _check_validcatlist(self, catlist):
        """
        Checking if all categorical covariates exist

        Parameters:
        ------------
        catlist: a list of categorical covariates

        """
        for cat in catlist:
            if cat not in self.data.columns:
                raise ValueError(f"{cat} cannot be found in the covariates")

    def _dummy_covar(self, catlist):
        """
        Converting categorical covariates to dummy variables

        Parameters:
        ------------
        catlist: a list of categorical covariates

        """
        covar_df = self.data[catlist]
        qcovar_df = self.data[self.data.columns.difference(catlist)]
        if (
            not qcovar_df.shape[1]
            == qcovar_df.select_dtypes(include=np.number).shape[1]
        ):
            raise ValueError("did you forget some categorical variables?")
        covar_df = pd.get_dummies(covar_df, drop_first=True, columns=catlist).astype(int)
        data = pd.concat([covar_df, qcovar_df], axis=1)

        return data

    def _add_intercept(self):
        """
        Adding the intercept

        """
        n = self.data.shape[0]
        # const = pd.Series(np.ones(n), index=self.data.index, dtype=int)
        # self.data = pd.concat([const, self.data], axis=1)
        self.data.insert(0, "intercept", np.ones(n))

    def _check_singularity(self):
        """
        Checking if a matrix is singular
        True means singular

        """
        if len(self.data.shape) == 1:
            return self.data == 0
        else:
            return np.linalg.cond(self.data) >= 1 / sys.float_info.epsilon


def get_common_idxs(*idx_list, single_id=False):
    """
    Getting common indices among a list of double indices for subjects.
    Each element in the list must be a pd.MultiIndex instance.

    Parameters:
    ------------
    idx_list: a list of pd.MultiIndex
    single_id: if return single id as a list

    Returns:
    ---------
    common_idxs: common indices in pd.MultiIndex or list

    """
    common_idxs = None
    for idx in idx_list:
        if idx is not None:
            if not isinstance(idx, pd.MultiIndex):
                raise TypeError("index must be a pd.MultiIndex instance")
            if common_idxs is None:
                common_idxs = idx.copy()
            else:
                common_idxs = common_idxs.intersection(idx)
    if common_idxs is None:
        raise ValueError("no valid index provided")
    if len(common_idxs) == 0:
        raise ValueError("no common index exists")
    if single_id:
        common_idxs = common_idxs.get_level_values("IID").tolist()

    return common_idxs


def get_union_idxs(*idx_list, single_id=False):
    """
    Getting union of indices from a list of double indices for subjects.
    Each element in the list must be a pd.MultiIndex instance.
    All duplicated indices will be removed.

    Parameters:
    ------------
    idx_list: a list of pd.MultiIndex
    single_id: if return single id as a list

    Returns:
    ---------
    union_idxs: union of indices in pd.MultiIndex or list

    """
    union_idxs = None
    for idx in idx_list:
        if idx is not None:
            if not isinstance(idx, pd.MultiIndex):
                raise TypeError("index must be a pd.MultiIndex instance")
            if union_idxs is None:
                union_idxs = idx.copy()
            else:
                union_idxs = union_idxs.union(idx, sort=False)
    if union_idxs is None:
        raise ValueError("no valid index provided")
    if single_id:
        union_idxs = union_idxs.get_level_values("IID").tolist()

    return union_idxs


def remove_idxs(idx1, idx2, single_id=False):
    """
    Removing idx2 (may be None) from idx1
    idx1 must be a pd.MultiIndex instance.

    Parameters:
    ------------
    idxs1: a pd.MultiIndex of indices
    idxs2: a pd.MultiIndex of indices
    single_id: if return single id as a list

    Returns:
    ---------
    idxs: indices in pd.MultiIndex or list

    """
    if not isinstance(idx1, pd.MultiIndex):
        raise TypeError("index must be a pd.MultiIndex instance")
    if idx2 is not None:
        idx = idx1.difference(idx2)
        if len(idx) == 0:
            raise ValueError("no subject remaining after --remove")
    else:
        idx = idx1
    if single_id:
        idx = idx.get_level_values("IID").tolist()

    return idx


def read_keep(keep_files):
    """
    Extracting common subject IDs from multiple files
    All files are confirmed to exist
    Empty files are skipped without error/warning
    files w/ or w/o a header are ok
    Error out if no common IDs exist

    Parameters:
    ------------
    keep_files: a list of tab/white-delimited files

    Returns:
    ---------
    keep_idvs_: pd.MultiIndex of common subjects

    """
    for i, keep_file in enumerate(keep_files):
        if os.path.getsize(keep_file) == 0:
            continue
        _, compression = utils.check_compression(keep_file)

        try:
            keep_idvs = pd.read_csv(
                keep_file,
                sep="\s+",
                header=None,
                usecols=[0, 1],
                dtype={0: str, 1: str},
                compression=compression,
            )
        except ValueError:
            raise ValueError("two columns FID and IID are required")

        keep_idvs = pd.MultiIndex.from_arrays(
            [keep_idvs[0], keep_idvs[1]], names=["FID", "IID"]
        )
        if i == 0:
            keep_idvs_ = keep_idvs.copy()
        else:
            keep_idvs_ = keep_idvs_.intersection(keep_idvs)

    if len(keep_idvs_) == 0:
        raise ValueError("no subjects are common in --keep")

    return keep_idvs_


def read_remove(remove_files):
    """
    Removing subject IDs from multiple files
    All files are confirmed to exist
    Empty files are skipped without error/warning
    files w/ or w/o a header are ok

    Parameters:
    ------------
    remove_files: a list of tab/white-delimited files

    Returns:
    ---------
    remove_idvs_: pd.MultiIndex of common subjects

    """
    for i, remove_file in enumerate(remove_files):
        if os.path.getsize(remove_file) == 0:
            continue
        _, compression = utils.check_compression(remove_file)

        try:
            remove_idvs = pd.read_csv(
                remove_file,
                sep="\s+",
                header=None,
                usecols=[0, 1],
                dtype={0: str, 1: str},
                compression=compression,
            )
        except ValueError:
            raise ValueError("two columns FID and IID are required")

        remove_idvs = pd.MultiIndex.from_arrays(
            [remove_idvs[0], remove_idvs[1]], names=["FID", "IID"]
        )
        if i == 0:
            remove_idvs_ = remove_idvs.copy()
        else:
            remove_idvs_ = remove_idvs_.union(remove_idvs, sort=False)

    return remove_idvs_


def read_voxel(voxel_file):
    """
    Reading a list of one-based voxels

    Parameters:
    ------------
    voxel_file: a file of voxels without headers

    Returns:
    ---------
    voxel_list: a np.array of zero-based voxels (N, )

    """
    voxels = pd.read_csv(voxel_file, header=None, sep="\s+", usecols=[0])
    try:
        int(voxels.iloc[0, 0])
    except ValueError:
        raise ValueError("headers are not allowed in --voxels")
    voxel_list = (voxels[0] - 1).values

    return voxel_list


def parse_input(arg):
    """
    Parsing files for LD matrix/LDR gwas

    Parameters:
    ------------
    arg: prefix file(s), e.g.
    `ldmatrix/ukb_white_exclude_phase123_25k_sub_chr{1:22}_LD1`
    `ldmatrix/ukb_white_exclude_phase123_25k_sub_allchr_LD1`
    `ukb_hippocampus_{0:25}.glm.linear`

    Returns:
    ---------
    A list of parsed files

    """
    p0 = r"\{.*:.*\}"
    p1 = r"{(.*?)}"
    p2 = r"({.*})"
    
    parse = arg.split(",")
    output = list()

    for x in parse:
        match = re.search(p0, x)
        if match:
            file_range = re.search(p1, x).group(1)
            try:
                start, end = [int(x) for x in file_range.split(":")]
            except ValueError:
                raise ValueError(
                    (
                        "if multiple files are provided, "
                        "they should be specified using `{}`, "
                        "e.g. `prefix_{stard:end}_suffix`. "
                        "Both start and end are included."
                    )
                )
            if start > end:
                start, end = end, start
            files = [re.sub(p2, str(i), x) for i in range(start, end + 1)]
            output.extend(files)
        else:
            output.append(x)

    return output


def keep_ldrs(n_ldrs, bases=None, ldr_cov=None, ldr_sumstats=None, resid_ldrs=None):
    """
    Extracting a specific number of LDRs

    Parameters:
    ------------
    bases: a np.array
    ldr_cov: a np.array
    ldr_sumstats: a Sumstats instance
    resid_ldrs: a pd.DataFrame

    """
    if bases is not None:
        if bases.shape[1] < n_ldrs:
            raise ValueError("the number of bases less than --n-ldrs")
        else:
            bases = bases[:, :n_ldrs]
    if ldr_cov is not None:
        if ldr_cov.shape[0] < n_ldrs:
            raise ValueError(
                "the dimension of variance-covariance matrix of LDR less than --n-ldrs"
            )
        else:
            ldr_cov = ldr_cov[:n_ldrs, :n_ldrs]
    if ldr_sumstats is not None:
        if ldr_sumstats.n_files < n_ldrs:
            raise ValueError("LDRs in summary statistics less than --n-ldrs")
        else:
            ldr_sumstats.n_files = n_ldrs
    if resid_ldrs is not None:
        if resid_ldrs.shape[1] < n_ldrs:
            raise ValueError("LDRs less than --n-ldrs")
        else:
            resid_ldrs = resid_ldrs.iloc[:, :n_ldrs]

    return bases, ldr_cov, ldr_sumstats, resid_ldrs


def check_existence(arg, suffix=""):
    """
    Checking file existence

    """
    if arg is not None and not os.path.exists(f"{arg}{suffix}"):
        raise FileNotFoundError(f"{arg}{suffix} does not exist")
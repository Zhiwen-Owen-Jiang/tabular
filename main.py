import os
import time
import argparse
import traceback
import numpy as np
import input.dataset as ds
from tabular.utils import GetLogger, sec_to_str


VERSION = "0.0.1"
MASTHEAD = (
    "******************************************************************************\n"
)
MASTHEAD += "* Voxelwise analysis for tabular data\n"
MASTHEAD += f"* Version {VERSION}\n"
MASTHEAD += f"* Zhiwen Jiang and Hongtu Zhu\n"
MASTHEAD += (
    f"* Department of Biostatistics, University of North Carolina at Chapel Hill\n"
)
MASTHEAD += f"* GNU General Public License v3\n"
MASTHEAD += f"* Correspondence: owenjf@live.unc.edu, zhiwenowenjiang@gmail.com\n"
MASTHEAD += (
    "******************************************************************************\n"
)

parser = argparse.ArgumentParser(
    description=f"\n Voxelwise analysis for tabular data v{VERSION}"
)

common_parser = parser.add_argument_group(title="Common arguments")
make_mt_parser = parser.add_argument_group(
    title="Arguments specific to making a hail.MatrixTable for tabular data"
)
tabular_parser = parser.add_argument_group(
    title="Arguments specific to doing association analysis between LDRs and tabular data"
)
sumstats_parser = parser.add_argument_group(
    title="Arguments specific to organizing and preprocessing association summary statistics"
)
voxel_assoc_parser = parser.add_argument_group(
    title="Arguments specific to reconstructing voxel-level association results"
)

# module arguments
make_mt_parser.add_argument(
    "--make-mt", action="store_true", help="Making a hail.MatrixTable of tabular data."
)
tabular_parser.add_argument(
    "--ldr-assoc", action="store_true", help="Association analysis between LDRs and tabular data."
)
sumstats_parser.add_argument(
    "--sumstats",
    action="store_true",
    help="Organizing and preprocessing association summary statistics.",
)
voxel_assoc_parser.add_argument(
    "--voxel-assoc", action="store_true", help="Recovering voxel-level association results."
)

# common arguments
common_parser.add_argument("--out", help="Prefix of output.")
common_parser.add_argument(
    "--threads",
    type=int,
    help=(
        "number of threads. "
        "Supported modules: --sumstats, --voxel-assoc."
    ),
)
common_parser.add_argument(
    "--n-ldrs",
    type=int,
    help=(
        "Number of LDRs. Supported modules: --ldr-assoc, --voxel-assoc."
    ),
)
common_parser.add_argument(
    "--ldr-sumstats",
    help=(
        "Prefix of preprocessed LDR GWAS summary statistics. "
        "Supported modules: --voxel-assoc"
    ),
)
common_parser.add_argument(
    "--bases",
    help=(
        "Directory to functional bases. Supported modules:  --voxel-assoc."
    ),
)
common_parser.add_argument(
    "--ldr-cov",
    help=(
        "Directory to variance-covariance marix of LDRs. "
        "Supported modules:  --voxel-assoc."
    ),
)
common_parser.add_argument(
    "--keep",
    help=(
        "Subject ID file(s). Multiple files are separated by comma. "
        "Only common subjects appearing in all files will be kept (logical and). "
        "Each file should be tab or space delimited, "
        "with the first column being FID and the second column being IID. "
        "Other columns will be ignored. "
        "Each row contains only one subject. "
        "Supported modules: --make-mt, --ldr-assoc."
    ),
)
common_parser.add_argument(
    "--remove",
    help=(
        "Subject ID file(s). Multiple files are separated by comma. "
        "Subjects appearing in any files will be removed (logical or). "
        "Each file should be tab or space delimited, "
        "with the first column being FID and the second column being IID. "
        "Other columns will be ignored. "
        "Each row contains only one subject. "
        "If a subject appears in both --keep and --remove, --remove takes precedence. "
        "Supported modules: --make-mt, --ldr-assoc."
    ),
)
common_parser.add_argument(
    "--covar",
    help=(
        "Directory to covariate file. "
        "The file should be tab or space delimited, with each row only one subject. "
        "Supported modules: --ldr-assoc."
    ),
)
common_parser.add_argument(
    "--cat-covar-list",
    help=(
        "List of categorical covariates to include in the analysis. "
        "Multiple covariates are separated by comma. "
        "Supported modules: --ldr-assoc."
    ),
)
common_parser.add_argument(
    "--voxels", "--voxel",
    help=(
        "One-based index of voxel or a file containing voxels. "
        "Supported modules: --voxel-assoc."
    ),
)
common_parser.add_argument(
    "--ldrs",
    help=(
        "Directory to LDR file. "
        "Supported modules: --ldr-assoc."
    ),
)
common_parser.add_argument(
    "--tabular-mt",
    help=(
        "Directory to MatrixTable. "
        "Supported modules: --ldr-assoc."
    ),
)
common_parser.add_argument(
    "--spark-conf",
    help=(
        "Spark configuration file. "
        "Supported modules: --make-mt, --ldr-assoc."
    ),
)
common_parser.add_argument(
    "--sig-thresh",
    type=float,
    help=(
        "p-Value threshold for significance, "
        "can be specified in a decimal 0.00000005 "
        "or in scientific notation 5e-08. "
        "Supported modules: --voxel-assoc."
    ),
)

# arguments for tabular.py
tabular_parser.add_argument(
    "--ldr-col", help="One-based LDR indices. E.g., `3,4,5,6` and `3:6`, must be consecutive"
)
make_mt_parser.add_argument(
    "--tabular-txt", 
    help=(
        "Directory to tabular data in txt format. "
        "The file should be tab or space delimited, with each row only one subject."
    ),
)

# arguments for sumstats.py
sumstats_parser.add_argument(
    "--ldr-raw-sumstats",
    help=(
        "Directory to raw LDR association summary statistics files. "
        "Multiple files can be provided using {:}, e.g., `ldr_assoc{1:10}.txt`, "
        "or separated by comma, but do not mix {:} and comma together."
    ),
)


def check_accepted_args(module, args, log):
    """
    Checking if the provided arguments are accepted by the module

    """
    accepted_args = {
        "make_mt": {
            "out",
            "make_mt",
            "tabular_txt",
            "keep",
            "remove",
            "spark_conf",
        },
        "ldr_assoc": {
            "out",
            "ldr_assoc",
            "tabular_mt",
            "ldr_col",
            "n_ldrs",
            "ldrs",
            "keep",
            "remove",
            "covar",
            "covar_cat_list",
            "spark_conf",
        },
        "sumstats": {
            "out",
            "sumstats",
            "ldr_raw_sumstats",
        },
        "voxel_assoc": {
            "out",
            "voxel_assoc",
            "sig_thresh",
            "voxels",
            "ldr_sumstats",
            "n_ldrs",
            "ldr_cov",
            "bases",
            "threads",
        },
    }

    ignored_args = []
    for k, v in vars(args).items():
        if v is None or not v:
            continue
        elif k not in accepted_args[module]:
            ignored_args.append(k)
            setattr(args, k, None)

    if len(ignored_args) > 0:
        ignored_args = [f"--{arg.replace('_', '-')}" for arg in ignored_args]
        ignored_args_str = ", ".join(ignored_args)
        log.info(
            f"WARNING: {ignored_args_str} ignored by --{module.replace('_', '-')}."
        )


def split_files(arg):
    files = arg.split(",")
    for file in files:
        ds.check_existence(file)
    return files


def process_args(args, log):
    """
    Checking file existence and processing arguments

    """
    ds.check_existence(args.covar)
    ds.check_existence(args.ldrs)
    ds.check_existence(args.spark_conf)
    ds.check_existence(args.tabular_mt)

    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError("--n-ldrs must be greater than 0")

    if args.keep is not None:
        args.keep = split_files(args.keep)
        args.keep = ds.read_keep(args.keep)
        log.info(f"{len(args.keep)} subject(s) in --keep (logical 'and' for multiple files).")

    if args.remove is not None:
        args.remove = split_files(args.remove)
        args.remove = ds.read_remove(args.remove)
        log.info(f"{len(args.remove)} subject(s) in --remove (logical 'or' for multiple files).")

    if args.voxels is not None:
        try:
            args.voxels = np.array(
                [int(voxel) - 1 for voxel in ds.parse_input(args.voxels)]
            )
        except ValueError:
            ds.check_existence(args.voxels)
            args.voxels = ds.read_voxel(args.voxels)
        if np.min(args.voxels) <= -1:
            raise ValueError("voxel index must be one-based")
        log.info(f"{len(args.voxels)} voxel(s) in --voxels.")
    
    if args.sig_thresh is not None:
        if args.sig_thresh <= 0 or args.sig_thresh >= 1:
            raise ValueError("--sig-thresh should be greater than 0 and less than 1")


def main(args, log):
    dirname = os.path.dirname(args.out)
    if dirname != "" and not os.path.exists(dirname):
        raise ValueError(f"{os.path.dirname(args.out)} does not exist")
    if (
        + args.sumstats
        + args.voxel_assoc
        + args.ldr_assoc
        + args.make_mt
        != 1
    ):
        raise ValueError(
            (
                "must raise one and only one of following module flags: "
                "--sumstats, --voxel-assoc, --ldr-assoc, --make-mt"
            )
        )

    if args.sumstats:
        check_accepted_args("sumstats", args, log)
        import tabular.sumstats as module
    elif args.voxel_assoc:
        check_accepted_args("voxel_assoc", args, log)
        import tabular.voxelassoc as module
    elif args.ldr_assoc:
        check_accepted_args('ldr_assoc', args, log)
        import tabular.tabular as module
    elif args.make_mt:
        check_accepted_args('make_mt', args, log)
        import tabular.mt as module

    process_args(args, log)
    module.run(args, log)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out is None:
        args.out = "tabular"

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(""))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "main.py \\\n"
        options = [
            "--" + x.replace("_", "-") + " " + str(opts[x]) + " \\"
            for x in non_defaults
        ]
        header += "\n".join(options).replace(" True", "").replace(" False", "")
        header = header + "\n"
        log.info(header)
        main(args, log)
    except Exception:
        log.info(traceback.format_exc())
        raise
    finally:
        log.info(f"\nAnalysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")
import os
import h5py
import logging
import math
import numpy as np
import pandas as pd
import input.dataset as ds


def check_input(args, log):
    # required arguments
    if args.ldr_raw_sumstats is None:
        raise ValueError("--ldr-raw-sumstats is required")

    args.ldr_raw_sumstats = ds.parse_input(args.ldr_raw_sumstats)
    for file in args.ldr_raw_sumstats:
        ds.check_existence(file)
    

def map_cols():
    """
    Creating two dicts for mapping provided colnames and standard colnames

    Returns:
    ---------
    cols_map: keys are standard colnames, values are provided colnames
    cols_map2: keys are provided colnames, values are standard colnames

    """
    cols_map = dict()
    cols_map["VARIABLE"] = "variable"
    cols_map["N"] = "n_called"
    cols_map["EFFECT"] = "beta"
    cols_map["null_value"] = 0
    cols_map["SE"] = "standard_error"
    cols_map["Z"] = "t_stat"

    cols_map2 = dict()
    for k, v in cols_map.items():
        if v is not None and k != "null_value":
            cols_map2[v] = k

    return cols_map, cols_map2


def read_sumstats(prefix):
    """
    Reading preprocessed summary statistics and creating a Sumstats instance.

    Parameters:
    ------------
    prefix: the prefix of summary statistics file

    Returns:
    ---------
    a Sumstats instance

    """
    info_dir = f"{prefix}.info"
    sumstats_dir = f"{prefix}.sumstats"

    if not os.path.exists(info_dir) or not os.path.exists(sumstats_dir):
        raise FileNotFoundError(f"either .sumstats or .info file does not exist")

    file = h5py.File(sumstats_dir, "r")
    info = pd.read_csv(info_dir, sep="\t", engine="pyarrow")

    if info.shape[0] != file.attrs["n_variables"]:
        raise ValueError(
            (
                "summary statistics and the meta data contain different number of variables, "
                "which means the files have been modified"
            )
        )

    return Sumstats(file, info)


class Sumstats:
    def __init__(self, file, info):
        """
        Parameters:
        ------------
        file: opened HDF5 file
        info: a pd.DataFrame of variable info

        """
        self.n_variables = file.attrs["n_variables"]
        self.n_files = file.attrs["n_files"]
        self.n_blocks = file.attrs["n_blocks"]
        self.file = file
        self.info = info
        self.idxs = None
        self.change_sign = None

    def close(self):
        self.file.close()

    def data_reader(self, data_type, file_idxs, idxs, all_file=True):
        """
        Reading summary statistics in chunks, each chunk of 20 LDRs
        Two modes:
        1. Reading a batch of LDRs and a subset of sumstats as a generator
        2. Reading all LDRs and a small proportion of sumstats into memory

        Parameters:
        ------------
        data_type: data type including `both`, `beta`, and `z`
        file_idxs (r, ): numerical indices of association file to extract
        idxs (d, ): numerical/boolean indices of variables to extract
        all_file: if reading all association sumstats

        Returns:
        ---------
        A np.array or a generator of sumstats

        """
        n_blocks = math.ceil(len(file_idxs) / 20)
        remaining = self.n_files

        if all_file:
            if data_type == "z":
                z_array = np.zeros((len(idxs), self.n_files), dtype=np.float32)
                for block_idx in range(n_blocks):
                    if remaining >= 20:
                        z_array[:, block_idx * 20 : (block_idx + 1) * 20] = self.file[
                            f"z{block_idx}"
                        ][:][idxs]
                        remaining -= 20
                    else:
                        z_array[:, block_idx * 20 : self.n_files] = self.file[
                            f"z{block_idx}"
                        ][:, :remaining][idxs]
                return z_array
            else:
                raise ValueError("only z-score can be read for all LDRs")
        else:
            return self._data_reader_generator(data_type, idxs, n_blocks)

    def _data_reader_generator(self, data_type, idxs, n_blocks):
        """
        Reading data as a generator

        """
        remaining = self.n_files

        if data_type == "both":
            for block_idx in range(n_blocks):
                if remaining >= 20:
                    yield [
                        self.file[f"beta{block_idx}"][:][idxs],
                        self.file[f"z{block_idx}"][:][idxs],
                    ]
                    remaining -= 20
                else:
                    yield [
                        self.file[f"beta{block_idx}"][:, :remaining][idxs],
                        self.file[f"z{block_idx}"][:, :remaining][idxs],
                    ]
        elif data_type == "beta":
            for block_idx in range(n_blocks):
                if remaining >= 20:
                    yield self.file[f"beta{block_idx}"][:][idxs]
                    remaining -= 20
                else:
                    yield self.file[f"beta{block_idx}"][:, :remaining][idxs]
        else:
            raise ValueError("other data type is not supported")


class ProcessSumstats:
    def __init__(
        self, files, cols_map, cols_map2, out_dir,
    ):
        """
        Parameters:
        ------------
        files: a list of files
        cols_map: a dict mapping standard colnames to provided colnames
        cols_map2: a dict mapping provided colnames to standard colnames
        out_dir: output directory

        """
        self.files = files
        self.n_files = len(files)
        self.cols_map = cols_map
        self.cols_map2 = cols_map2
        self.out_dir = out_dir
        self.logger = logging.getLogger(__name__)

    def _create_dataset(self, n_variables):
        self.current_block_idx = -1
        self.current_empty_space = 0

        with h5py.File(f"{self.out_dir}.sumstats", "w") as file:
            file.attrs["n_variables"] = n_variables
            file.attrs["n_files"] = 0  # initialize as 0
            file.attrs["n_blocks"] = 0

    def _save_sumstats(self, beta, z, is_last_file):
        """
        Saving sumstats in blocks

        Parameters:
        ------------
        beta (n_variables, n_ldrs): a np.array of beta to save
        z (n_variables, n_ldrs): a np.array of z to save
        is_last_file: if it is the last file
        idx: index of the current LDR to save
        self.current_empty_space: the number of empty columns in the current block
        self.current_block_idx: index of the corrent block

        """
        idx = 0
        n_ldrs = beta.shape[1]

        with h5py.File(f"{self.out_dir}.sumstats", "r+") as file:
            # checking if the current block is full, if not fill it first
            if self.current_empty_space != 0:
                file[f"beta{self.current_block_idx}"][
                    :, -self.current_empty_space :
                ] = beta[:, : self.current_empty_space]
                file[f"z{self.current_block_idx}"][:, -self.current_empty_space :] = z[
                    :, : self.current_empty_space
                ]

                if n_ldrs > self.current_empty_space:
                    file.attrs["n_files"] += self.current_empty_space
                    idx = self.current_empty_space
                    self.current_empty_space = 0
                else:
                    self.current_empty_space -= n_ldrs
                    file.attrs["n_files"] += n_ldrs
                    return

            # save remaining data to new blocks
            chunk_size = np.min((beta.shape[0], 10000))

            for i in range(idx, n_ldrs, 20):
                self.current_block_idx += 1
                file.attrs["n_blocks"] += 1
                file.create_dataset(
                    f"beta{self.current_block_idx}",
                    # data=beta[:, i: i+20],
                    shape=(file.attrs["n_variables"], 20),
                    dtype="float32",
                    chunks=(chunk_size, 20),
                )
                file.create_dataset(
                    f"z{self.current_block_idx}",
                    # data=z[:, i: i+20],
                    shape=(file.attrs["n_variables"], 20),
                    dtype="float32",
                    chunks=(chunk_size, 20),
                )

                end = np.min((i + 20, n_ldrs))
                file[f"beta{self.current_block_idx}"][:, : end - i] = beta[:, i:end]
                file[f"z{self.current_block_idx}"][:, : end - i] = z[:, i:end]
                file.attrs["n_files"] += end - i
                self.current_empty_space = 20 - end + i

            # remove the zero columns in the last block
            if is_last_file and self.current_empty_space > 0:
                last_beta = file[f"beta{self.current_block_idx}"][
                    :, : -self.current_empty_space
                ]
                last_z = file[f"z{self.current_block_idx}"][
                    :, : -self.current_empty_space
                ]
                del file[f"beta{self.current_block_idx}"]
                del file[f"z{self.current_block_idx}"]

                file.create_dataset(
                    f"beta{self.current_block_idx}",
                    data=last_beta,
                    dtype="float32",
                    chunks=(chunk_size, last_beta.shape[1]),
                )
                file.create_dataset(
                    f"z{self.current_block_idx}",
                    data=last_z,
                    dtype="float32",
                    chunks=(chunk_size, last_z.shape[1]),
                )

    def _save_info(self, info):
        info.to_csv(
            f"{self.out_dir}.info", sep="\t", index=None, na_rep="NA", float_format="%.3e"
        )

    def process(self, is_valid=None, info=None):
        """
        Processing LDR association summary statistics.

        """
        if is_valid is None and info is None:
            self.logger.info(
                (
                    f"Reading and processing {self.n_files} LDR association summary statistics file(s). "
                    "Only the first file will be QCed ..."
                )
            )
            is_valid, info = self._qc()
            self._save_info(info)
            self.logger.info("Reading and processing remaining files ...")

        self._create_dataset(info.shape[0])
        is_last_file = False
        for i, file in enumerate(self.files):
            if i == self.n_files - 1:
                is_last_file = True
            self._read_save(is_valid, file, is_last_file)

        return is_valid, info
        
    def _qc(self):
        """
        Quality control using the first file

        Returns:
        ---------
        is_valid: boolean indices of valid variables
        info: metadata

        """
        data = self._read_info(self.files[0])
        info = data[["VARIABLE", "N"]]
        is_valid = np.ones(data.shape[0], dtype=bool)
        self._create_dataset(data.shape[0])

        return is_valid, info
    
    def _read_effct(self, file):
        """
        Reading effects and z scores from HEIG association results

        """
        data = pd.read_parquet(
            file,
            columns=["beta", "t_stat"],
            engine="pyarrow",
        )
        data = data.rename(self.cols_map2, axis=1)

        return data

    def _read_save(self, is_valid, file, is_last_file):
        """
        Reading, processing, and saving a batch of LDR association files

        Parameters:
        ------------
        is_valid: boolean indices of valid variables
        file: an association file
        is_last_file: if it is the last file

        """
        data = self._read_effct(file)
        data = data.loc[is_valid]
        beta_array = np.array(list(data["EFFECT"]))
        z_array = np.array(list(data["Z"]))
        self._save_sumstats(beta_array, z_array, is_last_file)

    def _read_info(self, file):
        """
        Reading association info from a parquet file

        """
        data = pd.read_parquet(
            file,
            columns=["variable", "n_called"],
            engine="pyarrow",
        )
        data = data.rename(self.cols_map2, axis=1)

        return data


def run(args, log):
    check_input(args, log)
    cols_map, cols_map2 = map_cols()

    sumstats = ProcessSumstats(
            args.ldr_raw_sumstats, cols_map, cols_map2, args.out,
        )
    sumstats.process()

    log.info(f"\nSaved the processed summary statistics to {args.out}.sumstats")
    log.info(f"Saved the summary statistics information to {args.out}.info")
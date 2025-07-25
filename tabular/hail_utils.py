import os
import logging
import time
import shutil
import json
import hail as hl
import numpy as np
import pandas as pd


def get_temp_path(outpath):
    """
    Generating a path for saving temporary files

    """
    temp_path = outpath + "_temp"
    i = np.random.choice(1000000, 1)[0]  # randomly select a large number
    temp_path += str(i)

    return temp_path


def clean(out):
    if os.path.exists(out + "_spark"):
        shutil.rmtree(out + "_spark")
    if os.path.exists(out + "_tmp"):
        for _ in range(100):  # Retry up to 100 times
            try:
                shutil.rmtree(out + "_tmp")
                break  # Break if successful
            except OSError as e:
                time.sleep(1)  # Wait and retry


def init_hail(spark_conf_file, out):
    """
    Initializing hail

    Parameters:
    ------------
    spark_conf_file: spark configuration in json format
    out: output directory

    """
    with open(spark_conf_file, "r") as file:
        spark_conf = json.load(file)

    if "spark.local.dir" not in spark_conf:
        spark_conf["spark.local.dir"] = out + "_spark"

    tmpdir = out + "_tmp"
    logdir = out + "_hail.log"
    hl.init(
        quiet=True,
        spark_conf=spark_conf,
        local_tmpdir=tmpdir,
        log=logdir,
        tmp_dir=tmpdir,
    )
    


class TabularProcessor:
    """
    Tabular data processor

    """
    def __init__(self, input_dir,):
        self.mt = hl.read_matrix_table(input_dir)
        self.n_sub = self.mt.count_cols()
        self.n_variables = self.mt.count_rows()
        self.count_non_missing()
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            (
                f"{self.n_sub} subjects and "
                f"{self.n_variables} variables in the data.\n"
            )
        )
    
    def subject_id(self):
        """
        Extracting subject ids

        Returns:
        ---------
        mt_ids: a list of subject ids

        """
        mt_ids = self.mt.s.collect()
        if len(mt_ids) == 0:
            raise ValueError("no subject remaining in the data")
        return mt_ids

    def annotate_cols(self, table, annot_name):
        """
        Annotating columns with values from a table
        the table is supposed to have the key 'IID'

        Parameters:
        ------------
        table: a hl.Table
        annot_name: annotation name

        """
        table = table.key_by("IID")
        annot_expr = {annot_name: table[self.mt.s]}
        self.mt = self.mt.annotate_cols(**annot_expr)

    def keep_remove_idvs(self, keep_idvs, remove_idvs=None):
        """
        Keeping and removing subjects

        Parameters:
        ------------
        keep_idvs: a pd.MultiIndex/list/tuple/set of subject ids
        remove_idvs: a pd.MultiIndex/list/tuple/set of subject ids

        """
        filtering = None
        if keep_idvs is not None:
            if isinstance(keep_idvs, pd.MultiIndex):
                keep_idvs = keep_idvs.get_level_values("IID").tolist()
            keep_idvs = hl.literal(set(keep_idvs))
            filtering = keep_idvs.contains(self.mt.s)

        if remove_idvs is not None:
            if isinstance(remove_idvs, pd.MultiIndex):
                remove_idvs = remove_idvs.get_level_values("IID").tolist()
            remove_idvs = hl.literal(set(remove_idvs))
            if filtering is None:
                filtering =  ~remove_idvs.contains(self.mt.s)
            else:
                filtering = filtering & (~remove_idvs.contains(self.mt.s))
            
        if filtering is not None:
            self.mt = self.mt.filter_cols(filtering)
            self.count_non_missing()
            self.mt = self.mt.filter_rows(self.mt.n_called > 0)
            self.n_sub = self.mt.count_cols()
            self.n_variables = self.mt.count_rows()
            self.logger.info(
                (
                    f"{self.n_sub} subjects and {self.n_variables} variables after filtering."
                )
            )

    def count_non_missing(self):
        self.mt = self.mt.annotate_rows(
            n_called = hl.agg.count_where(hl.is_defined(self.mt.value))
        )

    def cache(self):
        self.mt = self.mt.cache()
        self.logger.info("Caching the data in memory.")
import os
import time
import shutil
import json
import hail as hl
import numpy as np


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
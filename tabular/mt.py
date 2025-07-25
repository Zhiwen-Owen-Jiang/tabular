import pandas as pd
import hail as hl
from tabular import utils
from tabular.hail_utils import *



def check_input(args, log):
    if args.spark_conf is None:
        raise ValueError("--spark-conf is required")


def tabular_to_mt(input_path, temp_path):
    """
    Convert a tabular phenotype file to a Hail MatrixTable.

    Parameters:
    -----------
    input_path: str
        Path to the input tabular file.
    temp_path: str
        Path to save the interim files.

    Returns:
    --------
    mt: hl.MatrixTable
        The resulting MatrixTable with data as columns.
    
    """
    openfunc, compression = utils.check_compression(input_path)
    cols = utils.check_header(openfunc, compression, input_path)

    first = True
    for chunk in pd.read_csv(input_path, chunksize=10000, sep='\s+'):
        long_chunk = pd.melt(chunk, id_vars=['IID'], value_vars=cols, var_name='data', value_name='value')
        long_chunk = long_chunk.rename(columns={'IID': 's'})
        long_chunk.to_csv(temp_path, mode="a", header=first, index=False, sep="\t", na_rep="NA")
        first = False
    
    entries_ht = hl.import_table(temp_path, delimiter='\t', missing='NA', impute=True)
    entries_ht = entries_ht.annotate(s=hl.str(entries_ht.s))
    entries_ht = entries_ht.key_by('data', 's')
    mt = entries_ht.to_matrix_table(row_key=['data'], col_key=['s'])
    
    return mt


def run(args, log):
    check_input(args, log)
    try:
        init_hail(args.spark_conf, args.out)
        temp_path = get_temp_path(args.out)
        tabular_mt = tabular_to_mt(args.tabular_data, temp_path)
        tabular_mt.write(f"{args.out}.mt", overwrite=True)
            
        log.info(f"Saved MatrixTable of tabular data at {args.out}.mt")
    finally:
        clean(args.out)
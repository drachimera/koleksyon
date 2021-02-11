#collection of data handling and data manipulation functions
from os import listdir
from os.path import isfile, join
import pandas as pd


def save_parquet(df, mypath, chunk_size=500000):
    """
    Assuming that we are working on a large memory machine, we can save a large file to parquet from a dataframe using this function
    note that pandas has a method, 'to_parquet' that should be used if you want to save just one file.
    This method is to be used when you want to save multiple parquet files (i.e. to follow the hadoop standard)
    df - dataframe containing the data
    mypath - location on the filesystem to put the parquet files with prefix (e.g. /home/user/data/cars will make parquet files prefixed with 'cars')
    chunk_size - max number of records in an individual file
    """


def load_parquet(mypath, wildcard):
    #given the memory is small, this can convert a directory of parquet files into a pandas dataframe   
    #mypath = "/dtascfs/data_ian/step2/"
    #wildcard = "q1many.pq.chunk_"
    #example usage:
    #df = loadParquet("/dtascfs/data_ian/step2/", "q1many.pq.chunk_")

    pqfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) if wildcard in f]

    dftmp = None
    vertical_stack = None

    for f in pqfiles:
        dftmp = pd.read_parquet(mypath + f)
        if vertical_stack is not None:
            # Stack the DataFrames on top of each other
            vertical_stack = pd.concat([vertical_stack, dftmp], axis=0)
        else:
            vertical_stack = dftmp
    nrec = len(vertical_stack)
    print("Number of Records Loaded: " + str(nrec))
    return vertical_stack
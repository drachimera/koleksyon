#collection of data handling and data manipulation functions

#general
import pandas as pd
import numpy as np

#file/os/sys
from os import listdir
from os.path import isfile, join

#math/calculations
import math

#date/time
from datetime import date 
today = date.today()

##############################
## File Functions
##############################

def save_parquet(df, mypath, chunk_size=500000):
    """
    Assuming that we are working on a large memory machine, we can save a large file to parquet from a dataframe using this function
    note that pandas has a method, 'to_parquet' that should be used if you want to save just one file.
    This method is to be used when you want to save multiple parquet files (i.e. to follow the hadoop standard)
    df - dataframe containing the data
    mypath - location on the filesystem to put the parquet files with prefix (e.g. /home/user/data/cars will make parquet files prefixed with 'cars')
    chunk_size - max number of records in an individual file
    @return - the files it created (absolute path)
    In other cases, e.g. where we extract from large database, we will need to iterate over the dataset and save the chunk as we obtain them.
    """
    saved_files = []
    number_of_chunks = math.ceil( len(df) / chunk_size)
    #print(number_of_chunks)
    for id, df_i in  enumerate(np.array_split(df, number_of_chunks)):
        file_i = mypath + '_{id}.pq'.format(id=id)
        df_i.to_parquet(file_i)
        saved_files.append(file_i)
    return saved_files


def load_parquet(mypath, wildcard):
    """
    Given the memory is large enough, this can convert a directory of parquet files into a pandas dataframe   
    mypath = a directory that has >1 parquet file that represent the same dataset.  e.g. /data/pqcars/
    wildcard - a string that all parquet files in the directory begin with.  e.g. if the directory contains [cars_0.pq, cars_1.pq] then wildcard is cars_
    #example usage:
    #df = loadParquet("../data/pqcars/", "cars_")
    """
    pqfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) if wildcard in f]

    dftmp = None
    vertical_stack = None

    for f in pqfiles:
        #print(f)
        dftmp = pd.read_parquet(mypath + f)
        if vertical_stack is not None:
            # Stack the DataFrames on top of each other
            vertical_stack = pd.concat([vertical_stack, dftmp], axis=0)
        else:
            vertical_stack = dftmp
    nrec = len(vertical_stack)
    print("Number of Records Loaded: " + str(nrec))
    return vertical_stack


##############################
## Variable Related Functions
##############################

# Example useage:
# categorical_features, numeric_features = ll.get_features_by_type(df, ["y1", "y2"])
def get_features_by_type(df, targets):
    """
    df - the dataframe we want analyzed
    targets - list of strings with the thing we are trying to predict, targets are not returned as categorical variables or numeric features
    """
    categorical_features = df.select_dtypes(include=['object']).columns
    for target_name in targets:
        if target_name in categorical_features:
            categorical_features = categorical_features.drop(target_name)
    #print(categorical_features)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for target_name in targets:
        if target_name in numeric_features:
            numeric_features = numeric_features.drop(target_name)
    return list(categorical_features), list(numeric_features)


##############################
## Date Related Functions
##############################

def convert_dates(df, columns):
    """
    Most of the time when you read from a csv rather than a static typed datastructure, you have to 'cast' the columns as dates.
    This is just a simple vaneer on top of pandas for those that always forget the syntax to do this.
    """
    for column in columns:
        df[column] =  pd.to_datetime(df[column])
    return df

#example usage:
#import datetime as dt
#from datetime import date
#ll.calculateAge(date(1997, 2, 3))
#bds = "1984-06-19"
#bd = dt.datetime.strptime(bds, '%Y-%m-%d')
#ll.calculateAge(bd.date())
def calculateAge(born):  
    #convert birthdate to age
    if born is None:
        return -1  #cleanup error in the data for no birthday
    try:  
        birthday = born.replace(year = today.year) 
  
    # raised when birth date is February 29 
    # and the current year is not a leap year 
    except ValueError:  
        birthday = born.replace(year = today.year, 
                  month = born.month + 1, day = 1) 
  
    if birthday > today: 
        return today.year - born.year - 1
    else: 
        return today.year - born.year

def calculate_age(df, column):
    """
    Simple library function that uses the lambda, calculateAge, above to create an age column in a dataframe
    """
    #create a new column, 'age', calculated from a column above it
    df[column]= pd.to_datetime(df[column]) 
    df['age'] = df.apply(lambda x: calculateAge(x[column]), axis=1)
    return df



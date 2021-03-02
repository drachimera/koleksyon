# koleksyon's lib interface provides several easy to use functions for understanding the distribution of data

#system
import os
import tempfile

#math
import math
import statistics
from scipy.stats import kurtosis, skew
from scipy.stats import uniform
from scipy.stats import norm

#ml
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#data
import numpy as np
import pandas as pd
import koleksyon.dta as dd

#plot
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image, SVG, display
from IPython.display import set_matplotlib_formats

#  listOfImageNames = ['/path/to/images/1.png',
#                      '/path/to/images/2.png']
#
def notebook_display_image_svg(listOfImageNames):
    """
    Takes as input a list of images (e.g. png) and displays them in the current cell
    """
    for imageName in listOfImageNames:
        display(SVG(filename=imageName))


def findMiddle(input_list):
    """
    Finds the middle element in a list.  if list is even, returns two middle elements
    """
    middle = float(len(input_list))/2
    l = []
    if middle % 2 != 0:
        l.append(input_list[int(middle - .5)])
        return l
    else:
        l.append(input_list[int(middle)])
        l.append(input_list[int(middle-1)])
        return l


#usage:
# a = [1,1,2,2,3]
# print(find_max_mode(a)) # print 2
def find_mode_mode(list1):
    """
    If you use statistics.mode on a list with multiple modes, it throws a math exception... which is a pain to 
    resolve when all you are trying to do is get some summary statistics on a variable.
    This will force the function to return a single mode if there are ties, and that single mode will be as
    close to the middle of the list as possible.  If the number is odd, it will be the middle, if even, we will pick the max of the two in the middle
    """
    list_table = statistics._counts(list1) #occurance/count pairs that tie for mode e.g. [(1, 2), (2, 2)]
    len_table = len(list_table)

    if len_table == 1:  #one value is the most common in the list, use that!
        mode_mode = statistics.mode(list1) #just one possibility, use it!
        return mode_mode
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        middle = findMiddle(new_list) # get the mode in the middle of the list
        if len(middle) == 1:
            return middle[0]
        else:
            return max(middle)


class dist_report():
    """
    goes to all of the various statistical libraries getting standard statistical measures, calculates them and holds them in an object.
    if you want to just get the report do:
        report = dist_report(df, 'variable')
        print(report)
    """
    def __init__(self, df, vname):
        self.vname = str(vname)
        x = df[vname]
        self.num = len(x)
        self.mn = min(x)
        self.mx = max(x)
        self.ave = np.mean(x)
        self.md = find_mode_mode(x)
        self.sd = np.std(x)
        self.kurt = kurtosis(x)
        self.skew = skew(x)
    def __str__(self):
        srep = ""
        srep = srep + "Statistics for  Variable:\t" + str(self.vname) + "\n"
        srep = srep + "Number of Data Points:\t" + str(self.num) + "\n"
        srep = srep + "Min:\t" + str(self.mn) + "\n"
        srep = srep + "Max:\t" + str(self.mx) + "\n"
        srep = srep + "Mean:\t" + str(self.ave) + "\n"
        srep = srep + "Mode:\t" + str(self.md) + "\n"
        srep = srep + "Variance:\t" + str(self.sd) + "\n"
        srep = srep + "Excess kurtosis of normal distribution (should be 0):\t" + str(self.sd) + "\n"
        srep = srep + "Skewness of normal distribution (should be 0):\t" + str(self.skew) + "\n"
        return srep



def density_plot(x, nplot=500, save_file="/tmp/density_plt.svg"):
    """
    plot the data in x in a density plot.
    x - list of the data points
    nplot - number of data points to take in the plot (helps see skewed data if we don't plot all of it)
    save_file - a temporary file where we place the image - the path to this is what is returned
    """
    #TODO:  figure out if I need to manipulate the output further to make it look better
    #sns.set(color_codes=True)
    #sns.set(rc={'figure.figsize':(10,10)})
    if os.path.exists(save_file):
        os.remove(save_file)
    subst = x.head(n=nplot) #x.head(n=45435)
    set_matplotlib_formats('png', 'pdf')
    # Density Plot and Histogram of all arrival delays
    dplot = sns.displot(subst.tolist(), kde=True, 
             bins=int(50), color = 'darkblue')
    plt.savefig(save_file, format="svg")
    return save_file


#exampe usage:
#var_analysis(df, 'difference', nplot=100)
def var_analysis(df, vname, nplot=500, save_file="/tmp/density_plt.svg"):
    """
    Do an in depth analysis of a single varible
    df - a pandas dataframe
    vname - variable to do the analysis on
    nplot - number of data points to include in plot
        nplot is mostly so that unbalanced histograms don't explode out of control
    This is just a pretty veneer on top of the other functions above, so not tested, but probably what most will use
    """
    x = df[vname]
    report = dist_report(df, vname)
    print(report)
    svg_file = density_plot(x, nplot, save_file)
    render_me = [svg_file]
    #notebook_display_image_svg(render_me)


#TODO: test me!
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

#TODO: test me!
#TODO: this is a bad name for what it is doing... probably should be named imputeTTSplit - for impute and remove NAs then do the scikit learn stuff...
#Utility function for creating training/testing data from a dataframe
#def data_prep(df2, target_variable):
#    df2 = clean_dataset(df2)  #TODO: there is probably a better way to deal with null data, imputation for example, need to put this in the lib
#    y = df2[target_variable]
#    X = df2.drop([target_variable], axis=1)
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#    return X_train, X_test, y_train, y_test

def data_prep(df, target_variable, missing_strategy="impute", balance_strategy="none", test_size=0.2):
    """
    Prepare the data for machine learning.
    df - original pandas dataframe
    target_variable - y, or the thing we are trying to predict.
    missing_strategy - how should we handle missing data / NA's?
        none=don't do anything
        droprow=drop all the rows with na,
        dropcol=drop all the columns with na, 
        fillna=fill all missing with -1, 
        fillave=fill missing columns with the average value for the column
        impute=sklearn.SimpleImputer, 
        iterative=sklearn.IterativeImputer
    balance_strategy - when the dataset is inbalanced, how do we try to correct it?
        none=don't do anything
        smote=apply smote to try to generate rows to balance the dataset
    test_size - the % of data you want in training vs testing... note, using 0 is a good hack for production... as it will do the same data prep operations
    """
    df2 = df
    if missing_strategy is "droprow":
        df2 = df.dropna()
    if missing_strategy is "dropcol":
        df2 = df.dropna()
    if missing_strategy is "fillna":
        df2 = df.fillna(-1)
    #do the multi-line scikit data prep
    
    if target_variable in df2.columns:  
        y = df2[target_variable]
        X = df2.drop([target_variable], axis=1)
        if(test_size == 1.0):
            return None, X, None, y
        elif(test_size == 0.0):
            return X, None, y, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            return X_train, X_test, y_train, y_test
    else:  #production data won't have a target, so just return what we can
        X = df2 
        y = None
        return X, X, y, y 


#TODO: test me!
#original post: https://towardsdatascience.com/a-data-scientists-guide-to-python-modules-and-packages-9193a861c26b
#usage:
#scale_data_list = data.select_dtypes(include=['int64', 'float64']).columns
#scale_data_df = pr.scale_data(data, scale_data_list)
def scale_data(df, column_list):
    """Takes in a dataframe and a list of column names to transform
     returns a dataframe of scaled values"""
    df_to_scale = df[column_list]
    x = df_to_scale.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_to_scale = pd.DataFrame(x_scaled, columns=df_to_scale.columns)
    return df_to_scale

#TODO: test me!
#usage:
#reduce_uniques_dict = {'education' : 1000,'occupation' : 3000, 'native-country' : 100}
#reduce_uniques_df = ll.reduce_uniques(data, reduce_uniques_dict)
def reduce_uniques(df, column_threshold_dict):
    """Takes in a dataframe and a dictionary consisting
    of column name : value count threshold returns the original
    dataframe"""
    for key, value in column_threshold_dict.items():
            counts = df[key].value_counts()
            others = set(counts[counts < value].index)
            df[key] = df[key].replace(list(others), 'Others')
            return df

#TODO: Test me!
def accuracy(data_y, npred, test="TEST"):
    mse = mean_squared_error(data_y, npred)
    r2 = r2_score(data_y, npred)
    print("**" + test + "**")
    print(test + " mean squared error = " + str(mse))
    rmse = math.sqrt(mse)
    print(test + " sqrt( mean squared error ) = " + str(rmse))
    print(test + " r2 value = " + str(r2))
    return rmse, mse, r2

#TODO: Test me!
def load_keep_columns(file):
    with open(file) as f:
        lines = f.read().splitlines()
    return lines

#TODO: Test me!
def load_drop_columns(df, targets, file):
    keep_columns = load_keep_columns(file)
    categorical_features, numeric_features = dd.get_features_by_type(df, targets)
    drop = []
    keep = []
    for feature in categorical_features:
        if(feature in keep_columns):
            keep.append(feature)
        else:
            drop.append(feature)
    for feature in numeric_features:
        if(feature in keep_columns):
            keep.append(feature)
        else:
            drop.append(feature)
    return drop

#TODO: Test me!
def fix_uncommon_keys(df, field, threshold=1, replaceValue="-1"):
    """
    sometimes we get values in a column that are really really rare, and those keys hurt rather than help machine learning algorithms.  This replaces those keys with unknown (-1)
    """
    hm = df[field].value_counts().to_dict()
    allowed_vals = []
    for k, v in hm.items():
        if v > threshold:
            allowed_vals.append(k)

    f = ~(df[field].isin(allowed_vals))
    df.loc[f, field] = replaceValue
    return df 
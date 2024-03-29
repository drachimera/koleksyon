# koleksyon's lib interface provides several easy to use functions for understanding the distribution of data

#system
import os
import tempfile

#math
import math
import statistics
import collections
from scipy.stats import kurtosis, skew
from scipy.stats import uniform
from scipy.stats import norm

#ml
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#performance stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

#data
import numpy as np
import pandas as pd
import koleksyon.dta as dd

#plot
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, SVG, display
from IPython.display import set_matplotlib_formats


#TODO: test this function!
def variables_by_type(df, target_name="__undefined__"):
    """
    easy method to get back catigorical and numeric fields in the dataframe (I always forget the exact syntax) AND you always have to take out the target!
    """
    categorical_features = df.select_dtypes(include=['object']).columns
    if target_name in categorical_features:
        categorical_features = categorical_features.drop(target_name)
    #print(categorical_features)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    if target_name in numeric_features:
        numeric_features = numeric_features.drop(target_name)
    #print(numeric_features)
    return categorical_features, numeric_features

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

#originally in the statistics lib...
def _counts(data):
    # Generate a table of sorted (value, frequency) pairs.
    table = collections.Counter(iter(data)).most_common()
    if not table:
        return table
    # Extract the values with the highest frequency.
    maxfreq = table[0][1]
    for i in range(1, len(table)):
        if table[i][1] != maxfreq:
            table = table[:i]
            break
    return table

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
    list_table = _counts(list1) #occurance/count pairs that tie for mode e.g. [(1, 2), (2, 2)]
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
        self.number = len(x)
        self.min = min(x)
        self.max = max(x)
        self.mean = np.mean(x)
        self.mode = find_mode_mode(x)
        self.variance = np.std(x)
        self.kurtosis = kurtosis(x)
        self.skew = skew(x)
    def __str__(self):
        srep = ""
        srep = srep + "Statistics for  Variable:\t" + str(self.vname) + "\n"
        srep = srep + "Number of Data Points:\t" + str(self.number) + "\n"
        srep = srep + "Min:\t" + str(self.min) + "\n"
        srep = srep + "Max:\t" + str(self.max) + "\n"
        srep = srep + "Mean:\t" + str(self.mean) + "\n"
        srep = srep + "Mode:\t" + str(self.mode) + "\n"
        srep = srep + "Variance:\t" + str(self.variance) + "\n"
        srep = srep + "Excess kurtosis of normal distribution (should be 0):\t" + str(self.kurtosis) + "\n"
        srep = srep + "Skewness of normal distribution (should be 0):\t" + str(self.skew) + "\n"
        return srep

def _stat_results_to_df(results):
    """
    results is a hash computed in some way that contains:
    key - an index for the row (unique)
    value - dist_report objects
    @return - a dataframe of summary statistics for the results
    """
    keys = list(results.keys())
    resultdf_cols = list(vars(results[keys[0]]).keys())  #columns in the output array, objects of type dist_report have these members
    resultsdf = pd.DataFrame(columns=resultdf_cols)
    for key in results.keys():
        values_to_add = vars(results[key])
        #print(values_to_add)
        row_to_add = pd.Series(values_to_add, name=key)
        resultsdf = resultsdf.append(row_to_add)

    resultsdf[resultdf_cols] = resultsdf[resultdf_cols].apply(pd.to_numeric, errors='coerce')
    resultsdf = resultsdf.drop(['vname'], axis=1)
    resultsdf.reset_index(inplace=True)
    resultsdf = resultsdf.rename(columns = {'index':'column name'})
    #resultsdf.reset_index(drop=True, inplace=True)
    #resultsdf = resultsdf.to_numeric(errors='coerce')
    return resultsdf

def _calculate_summary_stats_numeric_columns(df):
    """
    Utility function that calculates the summary statistics for all numeric_features and puts the results into a dataframe
    df - the original dataframe
    """
    categorical_features, numeric_features = variables_by_type(df)
    #first build a hash of columns
    results = {}
    for col in numeric_features:
        #print(col)
        results[col] = dist_report(df, col)
    #second convert the hash into a dataframe
    resultsdf = _stat_results_to_df(results)
    return resultsdf

def build_groupby_df(df, category, numeric, fill=-1):  
    """
    Copies a dataframe, df, and then groups the numeric data in column 'numeric' by a given category 'category'
    df - pandas dataframe
    numeric - the numeric column, e,g, price, los, time, ect.
    category - the categorical column e.g. service desk, drg, surgeon
    fill - the value we should use to fill NA, NaN ect
    """# numeric variable, categorical variable
    count_col = category + " count"
    sum_col = category + " sum"
    df_copy = df.copy()  
    df_copy[category].fillna(str(fill), inplace=True)
    df_copy[category] = df_copy[category].astype("object")
    df_copy[numeric].fillna(fill, inplace=True)
    df_copy[count_col] = df_copy.groupby(by=[category])[category].transform("count")
    df_copy[sum_col] = df_copy.groupby(by=[category])[numeric].transform("sum")
    df_copy[category + " groupby"] = df_copy[category].apply(str) + "_(" + df_copy[count_col].apply(str) + "|" + df_copy[sum_col].apply(str) + ")"
    df_copy.sort_values(by=[count_col])
    return df_copy

def _calculate_summary_stats_groupby(df, category, numeric, fill=-1):
    """
    Utility function that calculates the summary statistics for all numeric_features and puts the results into a dataframe
    df - dataframe built by running build_groupby_df
    fill - the value we should use to fill NA, NaN ect
    """
    calculated_column = category + " groupby"  #calculated by build_groupby_df
    groupdf = build_groupby_df(df, category, numeric, fill)
    categories = list(groupdf[calculated_column].value_counts().index)
    group = {} # A group to hold all of the instances of a given category
    for cat in categories:
        group[cat] = [] #empty list for now
    #go through df, appending the value @numeric to group lists
    for i in range(len(groupdf)):
        rowdf = groupdf.iloc[i]
        #print(rowdf[calculated_column])
        #print(rowdf[numeric])
        group[rowdf[calculated_column]].append(rowdf[numeric])
    #calculate statistics
    results = {}
    for key in group:  #key is a calculated column value e.g. Biggin_(897|650891790.0) ... , value is the values in column 'numeric' for all instances of that key in the dataframe
        ldf = pd.DataFrame (group[key], columns=[key]) #turn the list into a dataframe so we can compute stats
        dr = dist_report(ldf, key)
        results[key] = dr
    return _stat_results_to_df(results)




def calculate_summary_stats(df, category=-1, numeric=-1, fill=-1):
    """
    Returns a pandas dataframe of the key summary statistics for a column, vname, for a given dataframe, df.
    summary statistics are defined in the dist_report
    df - the dataframe
    category - the variable we wish to use to segment the data (each segment results in multiple rows out)
    numeric - the variable we are calculating statistics on, if not provided we will calculate for all
    fill - the value we should use to fill NA, NaN ect
    """
    if category == -1: #they just want summary statistics for df
        return _calculate_summary_stats_numeric_columns(df)
    elif category == -1 and numeric == -1:
        print("Error: must specify both category and numeric value to do a groupby...")
        return None
    else:
        return _calculate_summary_stats_groupby(df, category, numeric)

    

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

#TODO: this belongs in encode, not lib!  encode is where we put all the data pre-proc functions!
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
    if missing_strategy == "droprow":
        df2 = df.dropna()
    if missing_strategy == "dropcol":
        df2 = df.dropna()
    if missing_strategy == "fillna":
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


class AccuracyStats:
    def __init__(self, model_type='regressor'):
        """
        AccuracyStats - bring together common sklearn statistics, for rapid reporting
        type = regressor/classifier
        """
        self.model_type = model_type
        self.stats = {}
    def __str__(self):
        """
        string representation of score statistics... great for putting in a model training loop!
        """
        return str(self.stats)
    def calculate_stats(self, y_test, y_pred):
        """
        easy to remember convience method
        """
        if self.model_type == 'regressor':
            return self.regression_stats(y_test, y_pred)
        elif self.model_type == 'classifier':
            return self.classification_stats(y_test, y_pred)
    def classification_stats(self, y_test, y_pred):
        """
        calculate commonly used statistics for regression analysis
        """
        self.accuracy_score = accuracy_score(y_test, y_pred)  #sklearn; accuracy = num correct predictions / total predictions
        self.stats['accuracy_score'] = self.accuracy_score
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        #self.stats['confusion_matrix'] = self.confusion_matrix --removing this option as we don't want to print this nested structure when we have it all broken out anyway
        self.true_positives = self.confusion_matrix[0][0]
        self.stats['true_positives'] = self.true_positives
        self.false_positives = self.confusion_matrix[0][1]
        self.stats['false_positives'] = self.false_positives
        self.false_negatives = self.confusion_matrix[1][0]
        self.stats['false_negatives'] = self.false_negatives
        self.true_negatives = self.confusion_matrix[1][1]
        self.stats['true_negatives'] = self.true_negatives
        self.f1_score = f1_score(y_test, y_pred, average='macro')
        self.stats['f1_score'] = self.f1_score
        self.precision = float(self.true_positives) / ( float(self.true_positives) + float(self.false_positives) )
        self.stats['precision'] = self.precision
        self.recall = float(self.true_positives) / ( float(self.true_positives) + float(self.false_negatives) )
        self.stats['recall'] = self.recall
        self.roc_auc = roc_auc_score(y_test, y_pred)
        self.stats['roc_auc'] = self.roc_auc

        #REF: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
        #REF: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

        return self.stats

        #if(alg_type == "classifier"):
        #    score = f1_score(y_test, y_pred, average='macro')
        #    print("F1 Score = " + str(score))
        #elif(alg_type == "regressor"):
        #    score = r2_score(y_test, y_pred)
        #    print("R2 Score = " + str(score)) 
    def classification_report(self, y_test, y_pred, targets):
        """
        wrapper method, I always forget where to import this from sklearn!
        """
        return classification_report(y_test, y_pred, target_names=targets)
    #TODO: need ability to plot ROC
    def regression_stats(self, y_test, y_pred):
        self.mean_squared_error = mean_squared_error(y_test, y_pred)
        self.stats['mean_squared_error'] = self.mean_squared_error
        self.mean_absolute_error = mean_absolute_error(y_test, y_pred)
        self.stats['mean_absolute_error'] = self.mean_absolute_error
        self.sqrt_mean_squared_error = math.sqrt(self.mean_squared_error)
        self.stats['sqrt_mean_squared_error'] = self.sqrt_mean_squared_error
        self.r2_score = r2_score(y_test, y_pred)
        self.stats['r2_score'] = self.r2_score
        return self.stats 


#TODO: We need a function that does a pretty print of the accuracy stats with a label, but this ain't it!  perhaps call it pprint?
#TODO: Note, that we can currently just print the object above... so most of the work is done
#def accuracy(data_y, npred, test="TEST"):
#    mse = mean_squared_error(data_y, npred)
#    r2 = r2_score(data_y, npred)
#    print("**" + test + "**")
#    print(test + " mean squared error = " + str(mse))
#    rmse = math.sqrt(mse)
#    print(test + " sqrt( mean squared error ) = " + str(rmse))
#    print(test + " r2 value = " + str(r2))
#    return rmse, mse, r2

#TODO: Test me!
def load_keep_columns(file):
    with open(file) as f:
        lines = f.read().splitlines()
    return lines

#TODO: Test me!
def load_drop_columns(df, targets, file):
    """
    Loads a dataframe with only a subset of the columns
    TODO: we probably need to delete this because this exists:
    df = pd.read_csv('data.csv', skipinitialspace=True, usecols=fields)
    #https://stackoverflow.com/questions/26063231/read-specific-columns-with-pandas-or-other-python-module
    """
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
def saveToPDF(pdfFile, figs):
    '''
    Description: saves all figures in figs to a PDF file named pdfFile
    
    @input: 
        pdfFile: full path of PDF file including name
        figs: list of figures to save
    '''
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdfFile)
    for fig in figs: 
        pdf.savefig( fig )
    pdf.close()

#TODO: Test me!
def getGroupings(listVals, groupSize=20):
    '''
    Description: Get the groupings of size groupSize needed for boxplot from listVals
    @input:
        listVals: sorted list of values which will be used to create the groupings
        groupSize: size of each group used in boxplot (i.e. 20 MS DRGs)
    @return:
        groupOfN: a dictionary with starting index of each group as key and all elements of the group as value
                    i.e groupOfN.get(0) = ["470.0(1114|2009)", "871.0(632|4126)", ...]
    '''
    groupOfN={}
    currList=[]
    endIndex = 0
    for ix, val in enumerate(listVals):
        if (ix!=0) and (ix%groupSize==0):
            groupOfN[ix-groupSize] = currList
            endIndex = ix-1
            currList=[]
        currList.append(val)

    groupOfN[endIndex+1] = currList
    return groupOfN

#TODO: Test me!
def boxplotPrep(df, group, y, groupSize=20):
    """
    group - the name of the column that we want to group by (this will be on the x axis of the boxplot).  e.g. DRG, ADMISSION_DIAGNOSIS, DEPARTMENT, ect...
    y - the value that will be on the y axis of the boxplot.  Usually a number! charges, LOS, bed count, ect...
    """
    #columns we will be adding to df
    new_column_count = group + "_count"  
    new_column_sum = group + "_sum" 
    new_column_label = group + "_label"
    df[new_column_count] = df.groupby(by=[group])[group].transform("count")
    df[new_column_sum] = df.groupby(by=[group])[y].transform("sum")
    df[new_column_label] = df[group].apply(str) + "(" + df[new_column_count].apply(str) + "|" + df[new_column_sum].apply(str) + ")"
    df = df.sort_values(by=[new_column_count])
    freq = df[new_column_label].value_counts()
    groupings = getGroupings(freq.index, groupSize)
    return df, freq, groupings
#original usuage...
#df, freq, groupings = boxplotPrep(df, "ADMISSION_DIAGNOSIS", "LOS")
#df

#TODO: Test me!
def plotAGroup(df_main, byCol, valCol, filterList, title, figSize=(15,9)):
    '''
    Description: creates a boxplot for a group and returns reference to figure 
    @input:
        df_main: source dataframe to plot the data
        byCol: column by which boxplots would be created (i.e. MS DRG)
        valCol: column with distribution values (i.e. Length of Stay, Count of cases etc)
        filterList: a list of values for which plot would be created. This is used to filter the main dataframe
                    before plotting
        title: title of the boxplot
        figSize: dimensions of the figure
    '''
    params = {'axes.titlesize':'9'}
    matplotlib.rcParams.update(params)
    fig1, ax1 = plt.subplots(1,1,figsize=figSize)

    df_temp = df_main[df_main[byCol].isin(filterList)]
    axes=df_temp.boxplot(column=[valCol], by=[byCol], showmeans=True, ax=ax1, fontsize=8, return_type='axes')
    _=ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    _=ax1.set_title(f"{title}")
    
    return fig1
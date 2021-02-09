# koleksyon's lib interface provides several easy to use functions for understanding the distribution of data
import math
import statistics

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew
from scipy.stats import uniform
from scipy.stats import norm


#usage:
# a = [1,1,2,2,3]
# print(find_max_mode(a)) # print 2
def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode 

#exampe usage:
#var_analysis(df, 'difference', nplot=100)
def var_analysis(df, vname, nplot=500):
    """
    Do an in depth analysis of a single varible
    df - a pandas dataframe
    vname - variable to do the analysis on
    nplot - number of data points to include in plot
        nplot is mostly so that unbalanced histograms don't explode out of control
    """
    x = df[vname]
    print("Statistics for  Variable: " + vname)
    num = len(x)
    print("Number of Data Points: " + str(num))
    mn = min(x)
    print("Min: " + str(mn))
    mx = max(x)
    print("Max: " + str(mx))
    ave = np.mean(x)
    print("Mean: " + str(ave))
    md = find_max_mode(x)
    print("Mode: " + str(md))
    sd = np.std(x)
    print("Variance: " + str(sd))
    print( 'Excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(x) ))
    print( 'Skewness of normal distribution (should be 0): {}'.format( skew(x) ))

    subst = x.head(n=nplot) #differences.head(n=45435)
    # Density Plot and Histogram of all arrival delays
    sns.distplot(subst.tolist(), hist=True, kde=True, 
             bins=int(50), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})

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

#data
import numpy as np
import pandas as pd

#plot
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image, display
from IPython.display import set_matplotlib_formats

#  listOfImageNames = ['/path/to/images/1.png',
#                      '/path/to/images/2.png']
#
def notebook_display_image(listOfImageNames):
    """
    Takes as input a list of images (e.g. png) and displays them in the current cell
    """
    for imageName in listOfImageNames:
        display(Image(filename=imageName))


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
        self.md = find_max_mode(x)
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
    if os.path.exists(save_file):
        os.remove(save_file)
    subst = x.head(n=nplot) #x.head(n=45435)
    set_matplotlib_formats('png', 'pdf')
    # Density Plot and Histogram of all arrival delays
    sns.displot(subst.tolist(), kde=True, 
             bins=int(50), color = 'darkblue')
    plt.savefig(save_file, format="svg")
    return save_file


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



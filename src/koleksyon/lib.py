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




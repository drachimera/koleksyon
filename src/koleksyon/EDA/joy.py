#!/usr/bin/env python

# note, I had all kinds of problems getting the joy plot to work outside of rendering an image.

# Joy Plot allows the density curves of different groups to overlap, 
# it is a great way to visualize the distribution of a larger number of groups in relation to each other. 
# It looks pleasing to the eye and conveys just the right information clearly. 
# It can be easily built using the joypy package which is based on matplotlib.

# !pip install brewer2mpl
import numpy as np
import pandas as pd
import matplotlib as mpl


# if in a notebook, do this:
# %matplotlib inline
#note: need to configure backend for this:
# https://matplotlib.org/faq/usage_faq.html?highlight=backend#what-is-a-backend
#mpl.use("SVG")


import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")


# Import Data
df = pd.read_csv("./vizdata/mpg_ggplot2.csv")

# !pip install joypy
import joypy

# Draw Plot
plt.figure(figsize=(16,10), dpi= 80)
fig, axes = joypy.joyplot(df, column=['hwy', 'cty'], by="class", ylim='own', figsize=(14,10))

# Decoration
plt.title('Joy Plot of City and Highway Mileage by Class', fontsize=22)
print("saving a file: joy.svg")
plt.savefig('joy.svg')

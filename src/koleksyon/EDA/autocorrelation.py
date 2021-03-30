#!/usr/bin/env python

# Autocorrelation (ACF) and Partial Autocorrelation (PACF) Plot
# The ACF plot shows the correlation of the time series with its own lags. 
# Each vertical line (on the autocorrelation plot) represents the correlation between the series and its lag 
# starting from lag 0. 
# The blue shaded region in the plot is the significance level. 
# Those lags that lie above the blue line are the significant lags.

# So how to interpret this?

# For AirPassengers, we see upto 14 lags have crossed the blue line and so are significant. 
# This means, the Air Passengers traffic seen up to 14 years back has an influence on the traffic seen today.

# PACF on the other had shows the autocorrelation of any given lag (of time series) against the current series, 
# but with the contributions of the lags-inbetween removed.


import numpy as np
import pandas as pd
import matplotlib as mpl
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

# if in a notebook, do this:
# %matplotlib inline

# Import Data
df = pd.read_csv('./vizdata/AirPassengers.csv')
#print(df)


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Draw Plot
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)
plot_acf(df.value.tolist(), ax=ax1, lags=50)
plot_pacf(df.value.tolist(), ax=ax2, lags=20)

# Decorate
# lighten the borders
ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)
ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)
ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)
ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)

# font size of tick labels
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
plt.show()
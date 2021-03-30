import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import pandas as pd

df = pd.read_csv("property_tax_report_2018.csv")
# df['CURRENT_PRICE'] = df.apply(lambda x: x['CURRENT_LAND_VALUE']+x['CURRENT_IMPROVEMENT_VALUE'], axis = 1)
# df['PREVIOUS_PRICE'] = df.apply(lambda x: x['PREVIOUS_LAND_VALUE']+x['PREVIOUS_IMPROVEMENT_VALUE'], axis = 1)

# filter properties built on or after 1900
df_valid_year_built = df.loc[df['YEAR_BUILT'] >= 1900] 
# retrieve PID, YEAR_BUILT and ZONE_CATEGORY only
df1 = df_valid_year_built[['PID', 'YEAR_BUILT','ZONE_CATEGORY']]
# create 3 dataframes for 3 different zone categories
df_A = df1.loc[df1['ZONE_CATEGORY'] == 'Industrial']
df_B = df1.loc[df1['ZONE_CATEGORY'] == 'Commercial']
df_C = df1.loc[df1['ZONE_CATEGORY'] == 'Historic Area']
# retrieve the PID and YEAR_BUILT fields only
df_A = df_A[['PID','YEAR_BUILT']]
df_B = df_B[['PID','YEAR_BUILT']]
df_C = df_C[['PID','YEAR_BUILT']]
# Count the number of properties group by YEAR_BUILT
df2A = df_A.groupby(['YEAR_BUILT']).count()
df2B = df_B.groupby(['YEAR_BUILT']).count()
df2C = df_C.groupby(['YEAR_BUILT']).count()

# create line plots for each zone category 
fig, ax = plt.subplots()
l0, = ax.plot(df2A, lw=2, color='k', label='Industrial')
l1, = ax.plot(df2B, lw=2, color='r', label='Commercial')
l2, = ax.plot(df2C, lw=2, color='g', label='Historic Area')
# Adjusting the space around the figure
plt.subplots_adjust(left=0.2)
# Addinng title and labels
plt.title('Count of properties built by year')
plt.xlabel('Year Built')
plt.ylabel('Count of Properties Built')

#create a list for each zone category line plot
lines = [l0, l1, l2]

# make checkbuttons with all plotted lines with correct visibility
rax = plt.axes([0.05, 0.4, 0.1, 0.15])
# get the labels for each plot
labels = [str(line.get_label()) for line in lines]
# get the visibility for each plot
visibility = [line.get_visible() for line in lines]
check = CheckButtons(rax, labels, visibility)

# function to show the graphs based on checked labels
def func(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()

# on click event call function to display graph
check.on_clicked(func)

plt.show()

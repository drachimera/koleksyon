"""
========================================
Create 2D bar graphs in different planes
========================================

Demonstrates making a 3D plot which has 2D bar graphs projected onto
planes y=0, y=1, etc.
"""

# Import libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Create figure object
fig = plt.figure()

# Get the current axes, creating one if necessary.
ax = fig.gca(projection='3d')

# Get the Property Tax Report dataset
# Dataset link: https://data.vancouver.ca/datacatalogue/propertyTax.htm
data = pd.read_csv('property_tax_report_2018.csv')

# Groupby Zone catrgory and Year built to seperate for each zone
newdata = data.groupby(['YEAR_BUILT','ZONE_CATEGORY']).agg('count').reset_index()

# Create list of years that are found in all zones that we want to plot
years = [1995,2000,2005,2010,2015,2018]

# Create list of Zone categoreis that we want to plot
categories = ['One Family Dwelling', 'Multiple Family Dwelling', 'Comprehensive Development']

# Plot bar plot for each category
for cat,z,c in zip(categories,[1,2,3],['r','g','b']):
    category = newdata[(newdata['ZONE_CATEGORY']==cat) & (newdata['YEAR_BUILT'].isin(years))]
    ax.bar(category['YEAR_BUILT'], category['PID'],zs=z, zdir='y', color=c, alpha=0.8)
    
# Set labels
ax.set_xlabel('Years')
ax.set_ylabel('Distance Scale')
ax.set_zlabel('Number of Properties')

# Create customized legends 
legend_elements = [Line2D([0], [0], marker='s', color='w', label='One Family Dwelling',markerfacecolor='r', markersize=10),
                  Line2D([0], [0], marker='s', color='w', label='Multiple Family Dwelling',markerfacecolor='g', markersize=10),
                   Line2D([0], [0], marker='s', color='w', label='Comprehensive Development',markerfacecolor='b', markersize=10)
                  ]
                   
# Make legend
ax.legend(handles=legend_elements, loc='best')

plt.show()
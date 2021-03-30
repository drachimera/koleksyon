'''
==============
3D lineplot
==============

Demonstration of a basic lineplot in 3D.
'''

# Import libraries
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Set the legend font size to 10
mpl.rcParams['legend.fontsize'] = 10

# Create figure object
fig = plt.figure()

# Get the current axes, creating one if necessary.
ax = fig.gca(projection='3d')

# Create data point to plot
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

# Plot line graph 
ax.plot(x, y, z, label='Parametric curve')

# Set default legend
ax.legend()

plt.show()
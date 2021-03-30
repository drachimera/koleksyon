'''
==============
Contour Plots
==============
Plot a contour plot that shows intensity
'''
# Import libraries
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

# Create figure object
fig = plt.figure()

# Get the current axes, creating one if necessary.
ax = fig.gca(projection='3d')

# Get test data
X, Y, Z = axes3d.get_test_data(0.05)

# Plot contour curves
cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm)

# Set labels
ax.clabel(cset, fontsize=9, inline=1)

plt.show()
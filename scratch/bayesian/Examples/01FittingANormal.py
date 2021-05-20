import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

obs_y = np.random.normal(0.5, 0.35, 2000)  #try making the number of datapoints larger or smaller!

with pm.Model() as normal_example:
    stdev = pm.HalfNormal('stdev', sd=0.05) #standard deviations are positive
    mu = pm.Normal('mu', mu=0.0, sd=0.05)   #try changing sd to be smaller (0.05) or larger (0.5)
    y= pm.Normal('y', mu=mu, sd=stdev, observed=obs_y)
    
    trace = pm.sample(10000)
    pm.traceplot(trace, ['mu', 'stdev'])
    plt.show()
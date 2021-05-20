import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

# This is a better version of the same example using instead a Dirichlet distribution


preal = [0.1, 0.2, 0.6, 0.1]
y = np.random.choice(4, 1000, p=preal)

with pm.Model():
    
    p = pm.Dirichlet('p', a=np.ones(4))

    pm.Categorical(name='y', p=p, observed=y)

    trace = pm.sample(draws=1000, tune=200)

pm.traceplot(trace)


plt.savefig('dirichlet.png')
print("trace saved as dirichlet.png")

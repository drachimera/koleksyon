import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

# This is the 'fixed' version of  categoricalDebug where the model converges



preal = [0.1, 0.2, 0.6, 0.1]
y = np.random.choice(4, 1000, p=preal)
print(y)
with pm.Model():
    probs = []
    for k in range(4):
        _p = pm.Beta(name='p%i' % k, alpha=1, beta=1)
        probs.append(_p)

    p = tt.stack(probs)
    p1 = pm.Deterministic('p', p/p.sum())
    pm.Categorical(name='y', p=p1, observed=y)

    trace = pm.sample(draws=10000, tune=4000)

pm.traceplot(trace)
plt.savefig('congerge.png')
print("trace saved as converge.png")

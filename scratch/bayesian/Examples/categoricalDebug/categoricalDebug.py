import numpy as np
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

# This is an example of a categorical variable in a baysian model.  
# this specific example suffers from 'non-convergence' 
# can you figure out why?

with pm.Model():

    y = np.random.choice(4, 1000, p=[0.1, 0.2, 0.6, 0.1])
    print(y)

    probs = []

    for k in range(4):

        _p = pm.Beta(name='p%i' % k, alpha=1, beta=1)
        probs.append(_p)

    p = tt.stack(probs)
    pm.Categorical(name='y', p=p, observed=y)

    trace = pm.sample(target_accept=0.9, draws=10000, tune=5000)
    pm.traceplot(trace)
    plt.savefig('noncongerge.png')
    print("trace saved as nonconverge.png")

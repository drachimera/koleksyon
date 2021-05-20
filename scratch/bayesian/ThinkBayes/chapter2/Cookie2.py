#!/usr/env python
import sys
sys.path.append("../lib/ThinkBayes2/")
from thinkbayes2 import Pmf

pmf = Pmf()
#Prior Dist
pmf.Set('Bowl1', 0.5)
pmf.Set('Bowl2', 0.5)
print(pmf)

#Update based on new data
pmf.Mult('Bowl1', 0.75)
pmf.Mult('Bowl2', 0.5)
print(pmf)

#Hypotheses are mutally exclusive and collectively exhaustive, we can renormalize!
pmf.Normalize()
print(pmf) #posterior distribution

#book covers a more complicated implementation, but it is the same as Monty

print("Finished")

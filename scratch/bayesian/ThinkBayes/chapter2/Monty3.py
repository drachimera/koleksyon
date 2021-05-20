#!/usr/env python
import sys
sys.path.append("../lib/ThinkBayes2/code/")
from thinkbayes2 import Pmf

class Monty(Pmf):
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()
    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()
    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0;
        elif hypo == 'A':
            return 0.5
        else:
            return 1

hypos = 'ABC'
pmf = Monty(hypos)
print(pmf)

#calling update
pmf.Update('B')
print("After Update: ")
print(pmf)

print("Finished")

#!/usr/env python
import sys
sys.path.append("../lib/ThinkBayes2/")
from thinkbayes2 import Suite

class MM(Suite):
    mix94 = dict(brown=30,
             yellow=20,
             red=20,
             green=10,
             orange=10,
             tan=10,
             blue=0)

    mix96 = dict(blue=24,
             green=20,
             orange=16,
             yellow=14,
             red=13,
             brown=13)

    #Lay out the two options
    hypoA = dict(bag1=mix94, bag2=mix96)
    hypoB = dict(bag1=mix96, bag2=mix94)

#link the hypo into the suite...
    def Likelihood(self, data, hypo):
        bag, color = data
        mix = self.hypotheses[hypo][bag]
        like = mix[color]
        return like

    hypotheses = dict(A=hypoA, B=hypoB)
    print(hypotheses)

print("#############################################")
s = MM('AB')
print("#############################################")
print(s)
print("#############################################")
s.Update(('bag1', 'yellow'))
print(s)
print("#############################################")
s.Update(('bag2', 'green'))
print(s)
s.Update(('bag2', 'blue'))
print(s)

print("Finished")

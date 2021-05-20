#!/usr/env python
import sys
sys.path.append("../lib/ThinkBayes2/")
from thinkbayes2 import Pmf

pmf = Pmf()
for x in [1,2,3,4,5,6]:
    pmf.Set(x, 1/6.0)
print(pmf)

words = Pmf()
f = open('alice.txt','r')
for line in f:
    for word in line.split():
        words.Incr(word.strip(), 1)
#print(words)
words.Normalize()
print(words)


print("Finished")

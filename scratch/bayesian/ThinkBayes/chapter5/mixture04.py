from __future__ import print_function, division
import sys
sys.path.append("../lib/ThinkBayes2/code/")
from thinkbayes2 import Suite, Pmf, SampleSum, MakeMixture
import thinkplot
from simulationDD02 import Die

pmf_dice = Pmf()
pmf_dice.Set(Die(4), 2)
pmf_dice.Set(Die(6), 3)
pmf_dice.Set(Die(8), 2)
pmf_dice.Set(Die(12), 1)
pmf_dice.Set(Die(20), 1)
pmf_dice.Normalize()
print(pmf_dice)

print("#################################################")
mix = Pmf()
for die, weight in pmf_dice.Items():
    for outcome, prob in die.Items():
        mix.Incr(outcome, weight*prob)

#Shorthand for above
#mix = MakeMixture(pmf_dice)
print(mix)

thinkplot.Hist(mix)
thinkplot.Save(root='bar',
               xlabel='Mixture over a set of dice',
               ylabel='Probability',
               formats=['pdf'])

print("Program Finished")

from __future__ import print_function, division
import sys
sys.path.append("../lib/ThinkBayes2/code/")
from thinkbayes2 import Suite, Pmf, SampleSum
import thinkplot
import simulationDD02

d6 = simulationDD02.Die(6)
three_exact = d6 + d6 + d6
best_attr_cdf = three_exact.Max(6)
best_attr_pmf = best_attr_cdf.MakePmf()

#plot
thinkplot.PrePlot(1)
thinkplot.Plot(best_attr_pmf)
thinkplot.Save(root='DD2',
               xlabel='Max Sum of 3 d6 over 6 attributes',
               ylabel='Probability',
               formats=['pdf'])

print("Program Finished")

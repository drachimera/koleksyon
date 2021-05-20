''' This contains code that demonstrates solutions to problems from
chapter 4 of Allen Downey's book 'Think Bayes'.
'''

from __future__ import print_function, division
import sys
sys.path.append("../lib/ThinkBayes2/code/")
from thinkbayes2 import Suite, Pmf
import thinkplot


class Euro(Suite):
    def __init__(self, hypos):
        Suite.__init__(self, hypos)

    # Learned how to spell Likelihood correctly
    def Likelihood(self, data, hypo):
        if data == 'H':
            return hypo / 100.0
        else:
            return 1.0 - (hypo / 100.0)


def main():
    euro = Euro(range(101))
    euro.label = "Posterior probability of various biases"

    for data in range(140):
        euro.Update('H')

    for data in range(110):
        euro.Update('T')

    euro.Print()

    print("Mean hypothesis = {}".format(euro.Mean()))
    print("Median hypothesis = {}".format(euro.Median()))
    print("Hypothesis with maximum likelihood = {}".format(str(euro.MaximumLikelihood())))
    print("90% credible interval = {}".format(str(euro.CredibleInterval())))
    # euro.Print()

    # Use Allen Downey's thinkplot module to create a graph
    thinkplot.PrePlot(1)
    thinkplot.Plot(euro, style='')
    thinkplot.Save(root='euro1',
                   xlabel='Bias of heads vs. tails',
                   ylabel='Probability',
                   formats=['pdf'])


if __name__ == "__main__":
    main()

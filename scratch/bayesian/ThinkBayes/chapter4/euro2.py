''' This contains code that demonstrates solutions to problems from
chapter 4 of Allen Downey's book 'Think Bayes'.
'''

from __future__ import print_function, division
import sys
sys.path.append("../lib/ThinkBayes2/code/")
from thinkbayes2 import Suite, Pmf
import thinkplot


class Euro(Suite):
    def __init__(self, hypos, triangle_prior=False):
        Suite.__init__(self, hypos)
        if triangle_prior:
            for x in range(0, 51):
                self.Set(x, x)
            for x in range(51, 101):
                self.Set(x, 100 - x)
            self.Normalize()

    # Learned how to spell Likelihood correctly
    def Likelihood(self, data, hypo):
        if data == 'H':
            return hypo / 100.0
        else:
            return 1.0 - (hypo / 100.0)


def summarize_posterior(suite):
    print("Mean hypothesis = {}".format(suite.Mean()))
    print("Median hypothesis = {}".format(suite.Median()))
    print("Hypothesis with maximum likelihood = {}".format(str(suite.MaximumLikelihood())))
    print("90% credible interval = {}".format(str(suite.CredibleInterval())))
    print()


def main():
    euro = Euro(range(101))
    euro.label = "Uniform prior"
    euro_triangleprior = Euro(range(101), triangle_prior=True)
    euro_triangleprior.label = "Triangle prior"

    for data in range(140):
        euro.Update('H')
        euro_triangleprior.Update('H')

    for data in range(110):
        euro.Update('T')
        euro_triangleprior.Update('T')

    print("Summary for uniform prior: ")
    summarize_posterior(euro)

    print("Summary for triangle prior: ")
    summarize_posterior(euro_triangleprior)

    # Use Allen Downey's thinkplot module to create a graph
    thinkplot.PrePlot(1)
    thinkplot.Plot(euro)
    thinkplot.Plot(euro_triangleprior)
    thinkplot.Save(root='euro2',
                   xlabel='Bias of heads vs. tails',
                   ylabel='Probability',
                   formats=['pdf'])


if __name__ == "__main__":
    main()

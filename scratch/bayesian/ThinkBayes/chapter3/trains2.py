"""This file contains example code for Chapter 3 of "Think Bayes",
by Allen B. Downey.  Uses Allen's ThinkBayes2 library.  Jake Kugel
"""

from __future__ import print_function, division
import sys
sys.path.append("../lib/ThinkBayes2/code/")
from thinkbayes2 import Suite, Pmf
import thinkplot


class Train(Suite):
    '''A class that represents probabilities for suite of hypotheses about the
    number of trains operated by a single company.  Each hypothesis in the
    suite is represented by a number.
    '''
    def __init__(self, hypos, alpha=None):
        Pmf.__init__(self, hypos)
        if alpha is not None:
            for hypo in hypos:
                self.Set(hypo, hypo**(-alpha))
            self.Normalize()


    def Likelihood(self, data, hypo):

        # If the data completely rules out a hypothesis, probability is 0
        if data > hypo:
            return 0.0
        else:
            return 1 / hypo


def main():

    # Create a new Train object with hypotheses 1 (company has one train)
    # through 1000 (company has 1000 trains)
    train_uniform = Train(range(1, 1001))
    train_uniform.label = "Trains with uniform prior"
    train_powerlaw = Train(range(1, 1001), alpha=1.0)
    train_powerlaw.label = "Trains with power-law prior"

    # update the probability mass function with new data (train #60)
    train_uniform.Update(60)
    train_powerlaw.Update(60)

    # train.Print()

    print("uniform prior - mean hypothesis: {}".format(train_uniform.Mean()))
    print("Powerlaw - mean hypothesis: {}".format(train_powerlaw.Mean()))

    # Use Allen Downey's thinkplot module to create a graph
    thinkplot.PrePlot(1)
    thinkplot.Pmf(train_uniform)
    thinkplot.Pmf(train_powerlaw)
    thinkplot.Save(root='trains2',
                   xlabel='Number of trains',
                   ylabel='Probability',
                   formats=['pdf'])


if __name__ == "__main__":
    main()


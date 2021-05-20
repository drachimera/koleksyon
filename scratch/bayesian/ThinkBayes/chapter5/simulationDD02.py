from __future__ import print_function, division
import sys
sys.path.append("../lib/ThinkBayes2/code/")
from thinkbayes2 import Suite, Pmf, SampleSum
import thinkplot

class Die(Pmf):
    def __init__(self, sides):
        Pmf.__init__(self)
        for x in range(1, sides+1):
            self.Set(x, 1)
        self.Normalize()

def main():
    print("Single Die:")
    d6 = Die(6)
    print(d6)

    #use thinkbayes to simulate
    dice = [d6] * 3
    three = SampleSum(dice, 5000)
    print("##################################")
    print("Three Die:")
    print(three)

    #use thinkbayes to enumerate
    three_exact = d6 + d6 + d6
    print("##################################")
    print("Exact Three Die:")
    print(three_exact)

    # Use Allen Downey's thinkplot module to create a graph
    thinkplot.PrePlot(1)
    thinkplot.Plot(three)
    thinkplot.Plot(three_exact)
    thinkplot.Save(root='DD1',
                   xlabel='Sum of 3 d6',
                   ylabel='Probability',
                   formats=['pdf'])

    print("Program Complete")

if __name__ == "__main__":
    main()

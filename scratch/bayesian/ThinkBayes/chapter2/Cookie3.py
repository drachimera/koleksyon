#refactoring of the code in 'Cookie2' to make it more general
import sys
sys.path.append("../lib/ThinkBayes2/")
from thinkbayes2 import Pmf

#A Cookie object is a Pmf that maps from hypotheses to their probabilities.
class Cookie(Pmf):
    #The __init__ method gives each hypothesis the same prior probability.
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()  
        self.mixes = {
            'Bowl 1':dict(vanilla=0.75, chocolate=0.25),
            'Bowl 2':dict(vanilla=0.5, chocolate=0.5),
        }

    #Update method that takes data as a parameter and updates the probabilities
    def Update(self, data):
        #loops through each hypothesis in the suite and multiplies its probability
        #by the likelihood of the data under the hypothesis,
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()  

    #likelihood of the data under the hypothesis, is computed by
    def Likelihood(self, data, hypo):
        #Likelihood uses mixes, which is a dictionary that maps from the name of a bowl to the mix of cookies in the bowl.
        mix = self.mixes[hypo]
        like = mix[data]
        return like
    

#there are two hypotheses:
hypos = ['Bowl 1', 'Bowl 2']
pmf = Cookie(hypos)

#update
pmf.Update('vanilla')

#print the posterior probability of each hypothesis
for hypo, prob in pmf.Items():
    print(hypo, prob)

print("************************************************************")

#code is more complex, but we can do this!
# that it generalizes to the case where we draw more than one cookie from the same bowl (with replacement):
dataset = ['vanilla', 'chocolate', 'vanilla']
for data in dataset:
    pmf.Update(data)

for hypo, prob in pmf.Items():
    print(hypo, prob)
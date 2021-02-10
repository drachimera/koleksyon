# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest

import pandas as pd
import koleksyon.lib as ll


class TestLib(unittest.TestCase):

    def test_find_max_mode(self):
        print("Testing Max Mode...")
        a = [1,1,2,2,3]
        mm = ll.find_max_mode(a)
        self.assertEqual(mm, 2)

    def test_dist_report(self):
        print("Testing Distribution Report...")
        df = pd.read_csv("../data/cars.csv")
        report = ll.dist_report(df, 'MSRP')
        #print(report)  #pretty!
        rt = str(report).split("\n")
        #print(rt)
        expected = [
            "Statistics for  Variable:	MSRP",
            "Number of Data Points:	11914",
            "Min:	2000",
            "Max:	2065902",
            "Mean:	40594.737032063116",
            "Mode:	2000",
            "Variance:	60106.5809259237",
            "Excess kurtosis of normal distribution (should be 0):	60106.5809259237",
            "Skewness of normal distribution (should be 0):	11.770504957244958",
            ""
        ]
        self.assertEqual(len(expected), len(rt))
        i = 0
        for i in range(0, len(expected)):
            self.assertEqual(expected[i], rt[i])

    def test_density_plot(self):
        print("Testing Density Plot...")
        df = pd.read_csv("../data/cars.csv")
        x = df['MSRP']
        pltFile = ll.density_plot(x)
        print(pltFile)    
        

    #def test_var_analysis(self):
    #    df = pd.read_csv("../data/cars.csv")
    #    print(df)
    #    ll.var_analysis(df, "MSRP")

if __name__ == '__main__':
    unittest.main()

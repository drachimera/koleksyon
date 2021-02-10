# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest

import pandas as pd
import hashlib  #should just be needed in testing to see if the contents of a generated file are correct
import koleksyon.lib as ll


def check_contents(md5, filepath, ignore):
    """
    md5 - the md5 sum calculated last time the data was validated as correct
    filepath - the location/file where the new data is, this is to be validated
    ignore - a list of regular expressions that should be thrown out, line by line in the comparison
    """
    # Open,close, read file and calculate MD5 on its contents 
    with open(filepath,"r",encoding='utf-8') as file_to_check:
        # read contents of the file
        data = ""
        lines = file_to_check.readlines()
        for line in lines:
            flag = True
            for re in ignore:
                if re in line:
                    flag = False  #exclude this line, it's a date or something and will prevent the md5 from working
            if flag:
                data = data + line + "\n" 
        #print(data)   
        # pipe contents of the file through
        md5_returned = hashlib.md5(data.encode('utf-8')).hexdigest()
        print("Checking Contents Via Hash:")
        print("Original: " + md5)
        print("Calculated: " + md5_returned)
        if md5 == md5_returned:
            return True  #md5 verified
        else:
            return False #md5 verification failed!

class TestLib(unittest.TestCase):

    def test_find_mode_mode(self):
        print("Testing Mode Mode...")
        a = [1,1,2,2,3,3]
        mm = ll.find_mode_mode(a)
        self.assertEqual(mm, 2)
        b = [1,2,3,3]
        self.assertEqual(ll.find_mode_mode(b), 3)
        c = [1,2,2,3,3]
        self.assertEqual(ll.find_mode_mode(c), 2)
        d = [1,1,2,3,3]
        self.assertEqual(ll.find_mode_mode(d), 1)
        e = [1,1,2,2,2,3,3]
        self.assertEqual(ll.find_mode_mode(e), 2)
        

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

        # Correct original md5
        original_md5 = '07e1b5ecbc2f03eb8a1e7dc3b586a751' 

        df = pd.read_csv("../data/cars.csv")
        x = df['MSRP']
        pltFile = ll.density_plot(x)
        print(pltFile)
        #check that the rendering is the same as what we expect for this data/variable
        self.assertTrue(check_contents(original_md5, pltFile, ["<dc:date>", "style=", "path clip-path=", "clipPath id=", "xlink:href"]))
            
        

    #def test_var_analysis(self):
    #    df = pd.read_csv("../data/cars.csv")
    #    print(df)
    #    ll.var_analysis(df, "MSRP")

if __name__ == '__main__':
    unittest.main()

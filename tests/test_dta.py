import unittest
import os
import shutil
import pandas as pd
import hashlib  #should just be needed in testing to see if the contents of a generated file are correct
import koleksyon.dta as dd

class TestLib(unittest.TestCase):

    def test_save_parquet(self):
        print("Testing: save_parquet")
        dir_path = "/tmp/pqcars"
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            pass # the directory was not there, just make it in next line
        os.mkdir(dir_path)
        df = pd.read_csv("../data/cars.csv")
        dd.save_parquet(df,"/tmp/pqcars/cars",chunk_size=1000)


if __name__ == '__main__':
    unittest.main()
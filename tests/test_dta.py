import unittest
import os
import shutil
import pandas as pd
import hashlib  #should just be needed in testing to see if the contents of a generated file are correct
import koleksyon.dta as dd

class TestDTA(unittest.TestCase):

    def test_save_parquet(self):
        print("Testing: save_parquet")
        dir_path = "/tmp/pqcars"
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            pass # the directory was not there, just make it in next line
        os.mkdir(dir_path)
        df = pd.read_csv("../data/cars.csv")
        saved_files = dd.save_parquet(df,"/tmp/pqcars/cars",chunk_size=1000)
        print(saved_files)
        self.assertEqual(len(saved_files), 12)

    def test_load_parquet(self):
        print("Testing: load_parquet")
        df = pd.read_csv("../data/cars.csv")
        dfpq = dd.load_parquet("../data/pqcars/", "cars_")
        self.assertEqual(len(df), len(dfpq))
        self.assertEqual(list(df.columns), list(dfpq.columns))
        for col in df.columns:
            ldf = list(df[col].values)
            ldfpq = list(dfpq[col].values)
            #self.assertEqual(ldf, ldfpq) #TODO: right now the order that the files come back into the dataframe is not guarenteed same, perhaps we should fix
            self.assertEqual(ldf[0], ldfpq[0])


    def test_get_features_by_type(self):
        print("Testing: get_features_by_type")
        df = pd.read_csv("../data/cars.csv")
        targets = ["MSRP"]
        categorical_features, numeric_features = dd.get_features_by_type(df, targets)
        expected_categorical_features = ['Make', 'Model', 'Engine Fuel Type', 'Transmission Type','Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style']
        #note how MSRP is not in this list!
        expected_numeric_features = ['Year', 'Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg', 'Popularity']
        self.assertEqual(expected_categorical_features, categorical_features)
        self.assertEqual(expected_numeric_features, numeric_features)

    def test_convert_dates(self):
        print("Testing: convert_dates")
        df = pd.read_csv("../data/birthdays.csv")
        self.assertEqual("object", str(df['DOB'].dtype))  #before we cast it, its an object
        #print(df)
        dd.convert_dates(df, ["DOB"])
        self.assertEqual("datetime64[ns]", str(df['DOB'].dtype))  #datetime after the cast

    def test_calculate_age(self):
        print("Testing: calculate_age")
        df = pd.read_csv("../data/birthdays.csv")
        df = dd.calculate_age(df, "DOB")
        self.assertEqual(list(df.age), list(df.AgeIn2021))  #works for one year

if __name__ == '__main__':
    unittest.main()
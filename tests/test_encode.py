import unittest
import pandas as pd
from pandas._testing import assert_frame_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce
import koleksyon.encode as ee
import numpy as np

np.random.seed(42)

column_names_imports85 = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status',
                        'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country','income']

class TestEncode(unittest.TestCase):

    def test_get_encoders(self):
        print("Testing method for getting encoders that we may wish to consider in modeling")
        encoder_list = ee.get_encoders()
        print(encoder_list)
        self.assertEqual(15, len(encoder_list))

    
    def test_create_basic_encode_pipeline(self):
        self.maxDiff = None
        print("Testing that we can create the plain vanilla pipeline that: a) encodes the data, b) runs an algorithm")
        df = pd.read_csv("../data/imports-85.data", names=column_names_imports85)
        target_name = 'income'
        ep = ee.EncodePipeline(df, target_name, alg_type="classifier")
        rfc = RandomForestClassifier(n_estimators=500, random_state = 42)
        pipe = ep.create_basic_encode_pipeline(ce.one_hot.OneHotEncoder, rfc)
        #print(pipe)
        #check that the basic encode pipeline structure is correct by traversing the object structure and comparing it to this string (no whitespace)
        expected_object_structure = """
        Pipeline(steps=[('preprocessor',
                ColumnTransformer(transformers=[('num',
                                                Pipeline(steps=[('imputer',
                                                                   SimpleImputer(strategy='median')),
                                                                  ('scaler',
                                                                   StandardScaler())]),
                                                  Index(['age', 'workclass', 'fnlwgt', 'marital-status', 'gender',
       'hours-per-week', 'native-country'],
      dtype='object')),
                                                 ('cat',
                                                  Pipeline(steps=[('imputer',
                                                                   SimpleImputer(fill_value='missing',
                                                                                 strategy='constant')),
                                                                  ('woe',
                                                                   OneHotEncoder())]),
                                                  Index(['education', 'educational-num', 'occupation', 'relationship', 'race',
       'capital-gain', 'capital-loss'],
      dtype='object'))])),
                ('classifier',
                 RandomForestClassifier(n_estimators=500, random_state=42))])
        """
        self.assertEqual("".join(expected_object_structure.split()), "".join(str(pipe).split()))
        


    def test_classification(self):
 #   def foobar(self):
        print("Testing the encoder evaluation API on a classification problem")
        #Original Data location
        #url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status',
                        'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country','income']
        df = pd.read_csv("../data/adult.data", names=column_names)
        df = df.head(n=100)  #takes forever to run if we don't do this
        target_name = "income"   # <=50K, >50K
        ep = ee.EncodePipeline(df, target_name, "classifier")
        results = ep.evaluate_encoders()
        print(results)
        expected = pd.read_csv("testing_data/classifier_encoder_results.csv", index_col=0)
        print(expected)
        assert_frame_equal(expected, results)


    def test_regression_example(self):
        print("Testing Regression Example....")
    #    url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
        column_names = ['symboling', 'normalized-losses', 'make', 'fuel-type','aspiration',
                    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
                    'wheel-base', 'length', 'width', 'height', 'curb-weight',
                    'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
                    'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
                    'city-mpg', 'highway-mpg', 'price']
        car_data = pd.read_csv("../data/imports85.csv")
        print(car_data)
        print("Setting Up Machine Learning Algorithm...")
        ep = ee.EncodePipeline(car_data, 'price', 'regressor')
        results = ep.evaluate_encoders()
        print(results)
        #results.to_csv("regressor_encoder_results.csv")
        expected = pd.read_csv("testing_data/regressor_encoder_results.csv", index_col=0)
        print(expected)
        assert_frame_equal(expected, results)
        #TODO: test that results are consistent across runs!
        #alg = RandomForestRegressor(n_estimators=500)
#    print("Getting Encoders for Evaluation...")
#    encoders = ee.get_encoders()
#    #remove some encoders you don't want... e.g. woe encoders don't work on regression!
#    encoders.remove(ce.woe.WOEEncoder)
#    print(encoders)
#    print("Evaluating the Various Encoders...")
#    ee.evaluate(car_data, 'price', encoders, alg, algtype)



#example usuage on public data:
#if __name__ == "__main__":
#    print("Doing a Classification Example...")
#    classification_example()
#    print("=========================================================================")
#    print("Doing a Regression Example...")
#    regression_example()

if __name__ == '__main__':
    unittest.main()
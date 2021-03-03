import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce
import koleksyon.encode as ee

class TestEncode(unittest.TestCase):

    def test_get_encoders(self):
        print("Testing method for getting encoders that we may wish to consider in modeling")
        encoder_list = ee.get_encoders()
        print(encoder_list)
        self.assertEqual(15, len(encoder_list))

    def test_classification(self):
        print("Testing the API on a classification problem")
        #Original Data location
        #url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num','marital-status',
                        'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country','income']
        adults_data = pd.read_csv("../data/imports-85.data", names=column_names)
        print("Setting Up Machine Learning Algorithm...")
        algtype = "classifier"
        alg = RandomForestClassifier(n_estimators=500)
        print("Getting Encoders for Evaluation...")
        encoders = ee.get_encoders()
        #print(encoders)
        ##optional, remove some encoders you don't want...
    #print("Evaluating the Various Encoders...")
    #TODO: API for this thing needs to change!
    #ee.evaluate(adults_data, 'income', encoders, alg, algtype)

def regression_example():
    print("Downloading Data....")
    url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
    column_names = ['symboling', 'normalized-losses', 'make', 'fuel-type','aspiration',
                    'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
                    'wheel-base', 'length', 'width', 'height', 'curb-weight',
                    'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system',
                    'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
                    'city-mpg', 'highway-mpg', 'price']
    car_data = pd.read_csv(url_data, names=column_names)
    print("Setting Up Machine Learning Algorithm...")
    algtype = "regressor"
    alg = RandomForestRegressor(n_estimators=500)
    print("Getting Encoders for Evaluation...")
    encoders = ee.get_encoders()
    #remove some encoders you don't want... e.g. woe encoders don't work on regression!
    encoders.remove(ce.woe.WOEEncoder)
    print(encoders)
    print("Evaluating the Various Encoders...")
    ee.evaluate(car_data, 'price', encoders, alg, algtype)


#example usuage on public data:
#if __name__ == "__main__":
#    print("Doing a Classification Example...")
#    classification_example()
#    print("=========================================================================")
#    print("Doing a Regression Example...")
#    regression_example()

if __name__ == '__main__':
    unittest.main()
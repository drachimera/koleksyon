# the inclusion of the tests module is not meant to offer best practices for
# testing in general, but rather to support the `find_packages` example in
# setup.py that excludes installing the "tests" package

import unittest

import pandas as pd
import hashlib  #should just be needed in testing to see if the contents of a generated file are correct
import koleksyon.lib as ll
import koleksyon.dta as dd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

#testing datasets
from sklearn.datasets import load_breast_cancer

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

#TODO: this function needs to be redone!  especially in light of the new encode library...            
#    def test_data_prep(self):
#        print("Testing data_prep:")
#        df = dd.load_parquet("../data/melbourne/", "melbourne_") 
#        #first, don't do anything to the data... should have same number of rows as original...
#        X_train, X_test, y_train, y_test = ll.data_prep(df, 'Price', missing_strategy='none', test_size=1.0)
#        self.assertEqual(len(df), len(X_test))
#        self.assertEqual(len(df), len(y_test))
#        X_train, X_test, y_train, y_test = ll.data_prep(df, 'Price', missing_strategy='droprow', test_size=1.0)
#        #notice how this removes rows...
#        self.assertEqual(1778, len(X_test))
#        self.assertEqual(1778, len(y_test))

    #def test_var_analysis(self):
    #    df = pd.read_csv("../data/cars.csv")
    #    print(df)
    #    ll.var_analysis(df, "MSRP")

    ######################################################################
    #
    # Accuracy Statistics Below...
    # note the goal of AccuracyStats is not to replace sklearn, 
    # just to make sure people remember to calculate a variety of different summary statistics when they evaluate models!
    # below we test:
    # * classifier
    # * regressor
    #
    ######################################################################

    #test based on: https://towardsdatascience.com/a-practical-guide-to-seven-essential-performance-metrics-for-classification-using-scikit-learn-2de0e0a8a040
    def test_AccuracyStats_classifier(self):
        #STEP 1: prep data
        br_cancer = load_breast_cancer()
        #note we could leverage the data prep functions in koleksyon to make this easier, but this is simpler for a test...
        X, y = br_cancer['data'], br_cancer['target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        #create various different algorithms to test the performance statistics on them
        knn_model = KNeighborsClassifier(n_neighbors=3)
        knn_model.fit(X_train, y_train)

        sgd_model = SGDClassifier(random_state=42)
        sgd_model.fit(X_train, y_train)

        log_model = LogisticRegression()
        log_model.fit(X_train, y_train)

        #create predictions...
        y_pred_knn = knn_model.predict(X_test)
        y_pred_sgd = sgd_model.predict(X_test)
        y_pred_log = log_model.predict(X_test)

        #calculate some statistics, one for each algorithm  (usually, we don't have multiple algorithms, we have multiple runs)
        astats_knn = ll.AccuracyStats('classifier')
        stats_knn = astats_knn.calculate_stats(y_test, y_pred_knn)
        astats_sgd = ll.AccuracyStats('classifier')
        stats_sgd = astats_sgd.calculate_stats(y_test, y_pred_sgd)
        astats_log = ll.AccuracyStats('classifier')
        stats_log = astats_log.calculate_stats(y_test, y_pred_log)

        #check the accuracy statistics are correct
        self.assertAlmostEqual(0.9590643274853801, stats_knn['accuracy_score'])  #or astats_knn.accuracy_score, both work
        self.assertAlmostEqual(0.9649122807017544, stats_sgd['accuracy_score'])
        self.assertAlmostEqual(0.9824561403508771, stats_log['accuracy_score'])

        #check the confusion matrix (TP/FP/TN/FN) is correct
        # TP=59  |  FP=4
        # FN=3   |  TN=105
        self.assertEqual(59, astats_knn.true_positives)  #or stats_knn['true_positives'], both work
        self.assertEqual(4, astats_knn.false_positives)
        self.assertEqual(3, astats_knn.false_negatives)
        self.assertEqual(105, astats_knn.true_negatives)
        # TP=61  |  FP=2
        # FN=4   |  TN=104
        self.assertEqual(61, astats_sgd.true_positives)
        self.assertEqual(2, astats_sgd.false_positives)
        self.assertEqual(4, astats_sgd.false_negatives)
        self.assertEqual(104, astats_sgd.true_negatives)
        # TP=62  |  FP=1
        # FN=2   |  TN=106
        self.assertEqual(62, astats_log.true_positives)
        self.assertEqual(1, astats_log.false_positives)
        self.assertEqual(2, astats_log.false_negatives)
        self.assertEqual(106, astats_log.true_negatives)

        #check the F1 statistics are correct
        self.assertAlmostEqual(0.9558709677419355, stats_knn['f1_score'])  #or astats_knn.f1_score, both work
        self.assertAlmostEqual(0.962543808411215, stats_sgd['f1_score'])
        self.assertAlmostEqual(0.9812122321919062, stats_log['f1_score'])

        #check the precision statistics are correct
        self.assertAlmostEqual(0.9365079365079365, stats_knn['precision'])  #or astats_knn.f1_score, both work
        self.assertAlmostEqual(0.9682539682539683, stats_sgd['precision'])
        self.assertAlmostEqual(0.9841269841269841, stats_log['precision'])

        #check the recall statistics are correct
        self.assertAlmostEqual(0.9516129032258065, stats_knn['recall'])  #or astats_knn.f1_score, both work
        self.assertAlmostEqual(0.9384615384615385, stats_sgd['recall'])
        self.assertAlmostEqual(0.96875, stats_log['recall'])

        #check area under the curve (auc)
        self.assertAlmostEqual(0.9543650793650794, stats_knn['roc_auc'])  #or astats_knn.f1_score, both work
        self.assertAlmostEqual(0.9656084656084655, stats_sgd['roc_auc'])
        self.assertAlmostEqual(0.9828042328042328, stats_log['roc_auc'])



if __name__ == '__main__':
    unittest.main()

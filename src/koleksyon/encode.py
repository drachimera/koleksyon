import sys
import warnings

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
import category_encoders as ce
import koleksyon.lib as ll

# IMPORTANT NOTE: A previous version of this code considered 'testing' and tried to set the seed.

#another useful site:
#https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark

# A common question on many machine learning problems is to determine how to encode your data
# encoding is most often the process of mapping a column with strings into a column with numbers
# those strings could have a small set of acceptable values, or they could have practically free text
# regardless, strings need to be turned into numbers before you can use many machine learning techniques

# This program determines, for a dataframe, df, a good encoder strategy.

# Special Notes:
# This is NOT a method for a full on neural network, encoding there may be a bit different depending on the problem
# https://stats.idre.ucla.edu/spss/faq/coding-systems-for-categorical-variables-in-regression-analysis/ ... nice resource

def get_encoders():
    encoder_list = [ce.backward_difference.BackwardDifferenceEncoder,
               ce.basen.BaseNEncoder,
               ce.binary.BinaryEncoder,
                ce.cat_boost.CatBoostEncoder,
                ce.hashing.HashingEncoder,
                ce.helmert.HelmertEncoder,
                ce.james_stein.JamesSteinEncoder,
                ce.one_hot.OneHotEncoder,
                ce.leave_one_out.LeaveOneOutEncoder,
                ce.m_estimate.MEstimateEncoder,
                ce.ordinal.OrdinalEncoder,
                ce.polynomial.PolynomialEncoder,
                ce.sum_coding.SumEncoder,
                ce.target_encoder.TargetEncoder,
                ce.woe.WOEEncoder
                ]
    return encoder_list

#TODO: make this routine more general
def train_test(df, target_name, max_samples=-1, etest_size=0.2):
    """
    Easy macro for building train-test split
    df - pandas dataframe
    target_name - the column name of the variable we are trying to predict
    max_samples - if -1, we take all rows, otherwise we take n rows where n is the number provided
    etest_size - default test size is 20%, pass something else if you want a different distribution
    testing - is this being run in a unit test?  we will need to set the seed so it behaves the same each time
    """
    # Build training/testing
    print("Building Train/Test dataset")
    if max_samples == -1:
        max_samples = len(df) + 1  #if -1, use the entire dataset
    X = df.drop(target_name, axis=1).head(max_samples)
    y = df[target_name].head(max_samples)
    le = preprocessing.LabelEncoder()
    label_encoder = le.fit(y)
    y = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=etest_size)
    return X_train, X_test, y_train, y_test


class EncodePipeline:
    def __init__(self, df, target_name, alg_type):
        """
        df - well formed dataframe containing the data in a table organized for conventional regression/classification
        target_name - name of the variable in df that we are going to attempt to predict
        alg_type: "regressor"/"classifier"/"unsupervised" - what type of problem are we solving?
        """
        self.df = df
        self.target_name = target_name
        self.alg_type = alg_type
        self.categorical_features, self.numeric_features = ll.variables_by_type(df, target_name)
    def get_basic_supervised_algorithm(self):
        """
        In benchmarking various encoding stratigies we need a basic algorithm that trains fast and runs fast.  This just returns random forest for the problem type.
        """
        if self.alg_type == "regressor":
            alg = RandomForestRegressor(n_estimators=1000)
        if self.alg_type == "classifier":
            alg = RandomForestClassifier(n_estimators=1000)
        return alg
    def create_basic_encode_pipeline(self, encoder, algorithm):
        """
        Call this function when you want a basic encode pipeline without doing a bunch of coding.  It will do the following:
        - Simple Imputation
        - Data Scaling
        - transformer (whatever is provided)
        - transformers
        - algorithm (provided): code needs to inherit from the sklearn estimator interface.  example of how to do this (for custom transformer): https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65
        @params:
        encoder - text string specifying the encoder (same as what is returned from 'get encoders')
        algorithm: an object... instance of an estimator
        @return - a scikit pipeline that you can call pipe.fit on
        """
        self.algorithm = algorithm
        self.encoder = encoder
        iencoder = encoder()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('woe', iencoder)])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
                ])

        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                            (self.alg_type, algorithm)])
        self.pipeline = pipe
        return pipe

    def evaluate_encoders(self):
        """
        Evaluate a specific pipeline on the dataframe, df - we have a link to it because of the constructor
        """
        results = {}
        print("*******************************************************************")
        print("Benchmarking Encoders...")
        print("*******************************************************************")
        print("Preparing Data...")
        #don't want to deal with the empty data nonsense
        df = self.df.fillna(-1)
        X_train, X_test, y_train, y_test = train_test(df, self.target_name)
        print("*******************************************************************")
        print("Building Simple Algorithm...")
        alg = self.get_basic_supervised_algorithm()
        print("Evaluating Encoders...")
        encoders = get_encoders()
        for encoder in encoders:
            print("*******************************************************************")
            print(encoder)
            pipeline = self.create_basic_encode_pipeline(encoder, alg)
            try:
                print("Training....")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = pipeline.fit(X_train, y_train)
                print("Predicting....")
                y_pred = model.predict(X_test)
                accuracy_stats = ll.AccuracyStats(self.alg_type)
                stats = accuracy_stats.calculate_stats(y_test, y_pred) #returns hashmap of stats
                print(stats)
                results[str(encoder)] = stats
            except ValueError:
                print("Encoder Does not work on this data, do not use: " + str(encoder))

        return self.results_to_df(results)  #convert the dict of objects into a dataframe...
    def results_to_df(self, results):
        #convert the results into a dataframe, easier for users than a hash of objects!
        keys = list(results.keys())
        resultdf_cols = results[keys[0]].keys()
        #print(resultdf_cols)
        resultsdf = pd.DataFrame(columns=resultdf_cols)
        for key in results.keys():
            values_to_add = results[key]
            row_to_add = pd.Series(values_to_add, name=key)
            resultsdf = resultsdf.append(row_to_add)
        return resultsdf
        
        

#TODO: Test me!
#TODO: go find all those other functions that select rare values based on the distribution...
def fix_uncommon_keys(df, field, threshold=1, replaceValue="-1"):
    """
    sometimes we get values in a column that are really really rare, and those keys hurt rather than help machine learning algorithms.  This replaces those keys with unknown (-1)
    """
    hm = df[field].value_counts().to_dict()
    allowed_vals = []
    for k, v in hm.items():
        if v > threshold:
            allowed_vals.append(k)

    f = ~(df[field].isin(allowed_vals))
    df.loc[f, field] = replaceValue
    return df 


def evaluate(df, target_name, encoders, algorithm, alg_type, max_samples=500): #note max_samples is really small!
    #target_name is the name of the target variable in the dataframe, e.g. the thing we are trying to predict
    #numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    #categorical_features = df.select_dtypes(include=['object']).columns
    categorical_features = df.select_dtypes(include=['object']).columns
    if target_name in categorical_features:
        categorical_features = categorical_features.drop(target_name)
    #print(categorical_features)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    if target_name in numeric_features:
        numeric_features = numeric_features.drop(target_name)
    #print(numeric_features)

    #don't want to deal with the empty data nonsense
    df = df.fillna(-1)

    X_train, X_test, y_train, y_test = train_test(df, target_name, max_samples, testing=self.testing)

    for encoder in encoders:
        print("Setting Up Pipeline for Encoder:")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('woe', encoder())])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
                ])

        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                            (alg_type, algorithm)])

        print("*******************************************************************")
        print(encoder)
        print("Training....")
        model = pipe.fit(X_train, y_train)

        print("Predicting....")
        y_pred = model.predict(X_test)

        score = 0
        if(alg_type == "classifier"):
            score = f1_score(y_test, y_pred, average='macro')
            print("F1 Score = " + str(score))
        elif(alg_type == "regressor"):
            score = r2_score(y_test, y_pred)
            print("R2 Score = " + str(score))
        print("*******************************************************************")
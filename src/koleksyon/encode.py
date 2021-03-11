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

def variables_by_type(df, target_name):
    """
    easy method to get back catigorical and numeric fields in the dataframe (I always forget the exact syntax) AND you always have to take out the target!
    """
    categorical_features = df.select_dtypes(include=['object']).columns
    if target_name in categorical_features:
        categorical_features = categorical_features.drop(target_name)
    #print(categorical_features)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    if target_name in numeric_features:
        numeric_features = numeric_features.drop(target_name)
    #print(numeric_features)
    return categorical_features, numeric_features

#TODO: make this routine more general
def train_test(df, target_name, max_samples=-1, etest_size=0.2):
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
    def __init__(self, df, target_name):
        """
        df - well formed dataframe containing the data in a table organized for conventional regression/classification
        target_name - name of the variable in df that we are going to attempt to predict
        """
        self.df = df
        self.target_name = target_name
        self.categorical_features, self.numeric_features = variables_by_type(df, target_name)
    def create_basic_encode_pipeline(self, encoder, algorithm, alg_type):
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
        alg_type: "regressor"/"classifier"
        @return - a scikit pipeline that you can call pipe.fit on
        """
        self.alg_type = alg_type
        self.algorithm = algorithm
        self.encoder = encoder

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('woe', encoder())])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
                ])

        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                            (alg_type, algorithm)])
        self.pipeline = pipe
        return pipe

    def evaluate_pipeline(self):
        """
        Evaluate a specific pipeline on the dataframe, df - we have a link to it because of the constructor
        """
        X_train, X_test, y_train, y_test = train_test(self.df, self.target_name)
        print("*******************************************************************")
        print(self.encoder)
        print("Training....")
        model = self.pipeline.fit(X_train, y_train)

        print("Predicting....")
        y_pred = model.predict(X_test)
        print("*******************************************************************")
        




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

    X_train, X_test, y_train, y_test = train_test(df, target_name, max_samples)

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
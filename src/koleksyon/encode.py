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

def train_test(df, target_name, max_samples):
    # Build training/testing
    print("Building Train/Test dataset")
    X = df.drop(target_name, axis=1).head(max_samples)
    y = df[target_name].head(max_samples)
    le = preprocessing.LabelEncoder()
    label_encoder = le.fit(y)
    y = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

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
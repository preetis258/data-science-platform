import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import numpy as np


class DataPreprocessing:
    def __init__(self,dataframe,target):
        self.dataframe=dataframe
        self.target=target

    def num_cat_feat(self):
        num = []
        cat = []
        for i in self.dataframe.columns:
            if self.dataframe[i].dtype == 'O' or self.dataframe[i].nunique() <= 10:
                cat.append(i)
            else:
                num.append(i)
        return num, cat



    def missing_value_treatment(self,
                                num_method=None,
                                cat_method=None):
        num, cat = self.num_cat_feat()
        for i in num:
            if num_method == 'Mean':
                self.dataframe[i].fillna(self.dataframe[i].mean(), inplace=True)
            elif num_method == 'Median':
                self.dataframe[i].fillna(self.dataframe[i].median(), inplace=True)
            elif num_method == 'Zero':
                self.dataframe[i].fillna(0, inplace=True)
        for i in cat:
            if cat_method == 'Mode':
                self.dataframe[i].fillna(self.dataframe[i].mode()[0], inplace=True)
            elif cat_method == 'Unknown':
                self.dataframe[i].fillna('unknown', inplace=True)

        return self.dataframe, (self.dataframe.isnull().sum().sort_values(ascending=True))

    def outlier_treatment(self, method='IQR'):
        # replacing the outliers value with iqr
        # only numerical col
        # outlier treatment
        num = self.num_cat_feat()[0]

        if method == 'IQR':
            for i in num:
                if i != self.target:
                    iqr = self.dataframe[i].quantile(0.75) -self.dataframe[i].quantile(0.25)
                    lower_range = self.dataframe[i].quantile(0.25) - (iqr * 1.5)
                    upper_range =self.dataframe[i].quantile(0.75) + (iqr * 1.5)
                    self.dataframe.loc[self.dataframe[i] >= upper_range, i] = upper_range
                    self.dataframe.loc[self.dataframe[i] <= lower_range, i] = lower_range

        if method == 'Percentile':
            for i in num:
                if i != self.target:
                    lower_range =self. dataframe[i].quantile(0.01)
                    upper_range = self.dataframe[i].quantile(0.99)
                    self.dataframe.loc[self.dataframe[i] >= upper_range, i] = upper_range
                    self.dataframe.loc[self.dataframe[i] <= lower_range, i] = lower_range

        return self.dataframe


    def independent_dependent_features(self):

        X = self.dataframe.drop(self.target, axis=1)
        y = self.dataframe[self.target]

        return X, y

    def independent_feature_encoding(self, X, method='one_hot'):
        # X = self.independent_dependent_features()[0]
        encoding_col = list(X.select_dtypes(include='O').columns)


        #one hot encoding
        if method == 'OneHot':
            # all the categoricals feature will be encoding with one hot encoding
            one_hot_enc = OneHotEncoder(sparse=False,handle_unknown='ignore')
            encoded_df = pd.DataFrame(one_hot_enc.fit_transform(X[encoding_col]),
                                      columns=one_hot_enc.get_feature_names_out())

        # ordinal encoding
        elif method == 'Ordinal':
            ordinal_encoder = OrdinalEncoder()
            encoded_df = pd.DataFrame(ordinal_encoder.fit_transform(X[encoding_col]),
                                     columns = encoding_col)

        X.drop(columns=encoding_col, inplace=True)
        X.reset_index(drop=True,inplace=True)

        X_encoded = pd.concat([X, encoded_df], axis=1)
        # X_encoded.reset_index(drop=True,inplace=True)

        return X_encoded, encoding_col

    def dependent_feature_encoding(self,y):
        # y = self.independent_dependent_features()[1]
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        return y_encoded






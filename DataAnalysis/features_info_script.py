

import numpy as np
import pandas as pd


class ColumnInformation:

    def __init__(self,dataframe):
        self.dataframe=dataframe

    def missing_value_features(self):
        # This function returns the list of features that have missing values and its count
        missing_df=self.dataframe.isnull().sum().sort_values(ascending=False).to_frame(name='Missing values count')
        missing_df=missing_df[missing_df['Missing values count'] > 0].reset_index().rename(columns={"index":"Columns"})
        return missing_df.shape[0], missing_df

    def num_cat(self):
        # this function return the list of numerical and categorical features
        numerical_features = []
        categorical_features = []
        for i in self.dataframe.columns:
            if self.dataframe[i].dtype == 'O' or self.dataframe[i].nunique() <= 5:
                categorical_features.append(i)
            else:
                numerical_features.append(i)
        return numerical_features, categorical_features


    def number_of_duplicates_records(self):
        # this function return the number of duplicate records
        # return self.num_cat()[1]
        return self.dataframe.duplicated().sum()


    def outier_features(self,method,min_val=None,max_val=None): #Inter Quartile Range (IQR)','Percentile'
        outlier_df=pd.DataFrame({'Features':[],'Outlier count':[],'Outlier Percentage':[],
                    'Lower Range':[],'Upper Range':[]})
        if method=='Inter Quartile Range (IQR)':
            outlier_count=0

            for i,col in enumerate(self.num_cat()[0]):
                q75, q25 = np.percentile(self.dataframe.dropna().loc[:, col], [75, 25])
                iqr = q75 - q25
                lower_range = q25 - (iqr*1.5)
                upper_range = q75 + (iqr*1.5)
                count=self.dataframe.loc[self.dataframe.loc[:col][col]< lower_range,col].count()+self.dataframe.loc[self.dataframe.loc[:col][col]> upper_range,col].count()
                if count>0:
                    outlier_count+=1
                percent_of_outlier=(count/len(self.dataframe)*100)
                outlier=pd.DataFrame({'Features':[col],'Outlier count':[count],'Outlier Percentage':[percent_of_outlier],
                    'Lower Range':[lower_range],'Upper Range':[upper_range]},index=[i])

                outlier_df=outlier_df.append(outlier)
            return outlier_count,outlier_df

        elif method=='Percentile':
            outlier_count = 0

            for i, col in enumerate(self.num_cat()[0]):
                p1, p99 = np.percentile(self.dataframe.dropna().loc[:, col], [min_val, max_val])
                lower_range = p1
                upper_range = p99
                count = self.dataframe.loc[self.dataframe.loc[:col][col] < lower_range, col].count() + self.dataframe.loc[self.dataframe.loc[:col][col] > upper_range, col].count()
                if count > 0:
                    outlier_count += 1
                percent_of_outlier = (count / len(self.dataframe) * 100)
                outlier = pd.DataFrame(
                    {'Features': [col], 'Outlier count': [count], 'Outlier Percentage': [percent_of_outlier],
                     'Lower Range': [lower_range], 'Upper Range': [upper_range]}, index=[i])

                outlier_df = outlier_df.append(outlier)


            return outlier_count, outlier_df



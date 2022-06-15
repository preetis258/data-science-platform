
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


class TargetInformation:
    def __init__(self,dataframe):
        self.dataframe = dataframe

# if target column == classification
    def target_classification(self,target_col):
        '''a function that takes classification target column and returns the basic informations'''
        unique_categories = self.dataframe[target_col].value_counts().index.tolist()
        class_sizes = list(self.dataframe[target_col].value_counts().values)
        return unique_categories, class_sizes

    def imbalance_checker(self,target_col):
        values = list(self.dataframe[target_col].value_counts(normalize=True).values*100)
        for idx in range(len(values)-1):
            if abs(values[idx]-values[idx+1])>=40:
                return values,True
            else:
                return values,False

    def plot_countplot_target(self,target_col):
        categories,values=self.target_classification(target_col)
        prct=self.imbalance_checker(target_col)[0]
        df_target=pd.DataFrame({target_col:categories,"Count":values,"Percentage":np.round(prct,2)})
        fig=px.bar(data_frame=df_target, x=target_col,y='Count', barmode="group",hover_name="Percentage",title=f'Countplot of {target_col}')
        return fig

    def target_regression(self, target_col):
        '''a function that takes regression target column and returns the basic informations'''
        stats=['Count','Mean','Median','Minimum value','Maximum value','Range','Variance','Standard deviation']
        count=self.dataframe[target_col].count()
        mean_val=self.dataframe[target_col].mean()
        median_val=self.dataframe[target_col].median()
        min_val=self.dataframe[target_col].min()
        max_val=self.dataframe[target_col].max()
        range_val=max_val-min_val
        variance_val=self.dataframe[target_col].var()
        std_val=self.dataframe[target_col].std()
        values=[count,mean_val,median_val,min_val,max_val,range_val,variance_val,std_val]
        return pd.DataFrame({"Statistic":stats,"Values":values})

    def plot_histplot_target(self,target_col,kde_curve=False,histogram=False):
        li = list(self.dataframe[target_col].dropna())
        fig = ff.create_distplot([li],group_labels=[target_col],show_curve=kde_curve,show_hist=histogram,show_rug=False)
        if (kde_curve==False)&(histogram==True):
            fig.update_layout(title=f'Histogram plot of {target_col}')
        elif (kde_curve==True)&(histogram==True):
            fig.update_layout(title=f'Histogram and distribution curve of {target_col}')
        else:
            fig.update_layout(title=f'Distribution curve of {target_col}')
        fig.update_layout( showlegend=False )
        return fig



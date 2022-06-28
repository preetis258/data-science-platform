
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


class Univariate_viz:
    def __init__(self,dataframe):
        self.dataframe=dataframe

    def features_segregation(self):
        cat_cols=self.dataframe.select_dtypes("O").columns.tolist()
        cat_col_10=[col for col in cat_cols if self.dataframe[col].nunique()<=10]
        cat_col_more_10=[col for col in cat_cols if col not in cat_col_10]
        num_cols=self.dataframe.select_dtypes(exclude='O').columns.tolist()
        num_cols_more_10=[]
        for col in num_cols:
            if self.dataframe[col].nunique()<=10:
                cat_cols.append(col)
                cat_col_10.append(col)
            else:
                num_cols_more_10.append(col)

        return cat_cols,cat_col_10,cat_col_more_10,num_cols_more_10

    def plot_categorical(self,col):
        '''function to plot count and pie plots for a categorical feature'''
        if col in self.features_segregation()[2]:
            values = [i for i in self.dataframe[col].value_counts().sort_values(ascending=False).head(10).values]
            label = [i for i in self.dataframe[col].value_counts().sort_values(ascending=False).head(10).index]
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'xy'}, {'type': 'domain'}]])
            fig.add_trace(go.Bar(x=label, y=values, name=f"{col} bar chart",marker_color='lightslategrey',showlegend=False), 1, 1)
            fig.add_trace(go.Pie(labels=label, values=values, name=f"{col}"), 1, 2)
            fig.update_layout(
                title_text=f"Distribution of {col} (Top 10)")
            # fig.update_layout(showlegend=False)
            return fig

        else:
            values = [i for i in self.dataframe[col].value_counts().values]
            label = [i for i in self.dataframe[col].value_counts().index]
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'xy'}, {'type': 'domain'}]])
            fig.add_trace(go.Bar(x=label,y=values,name=f"{col} bar chart",marker_color='lightslategrey',showlegend=False),1,1)
            fig.add_trace(go.Pie(labels=label, values=values, name=f"{col}"),1,2)
            fig.update_layout(
                title_text=f"Distribution of {col}")
            return fig


    def plot_numerical(self,col):
        '''function to plot histogram and distribution curve for a numerical feature'''
        self.dataframe=self.dataframe.dropna(subset=[col])
        fig = make_subplots(rows=1,cols=2)
        fig.add_trace(go.Histogram(x=self.dataframe[col], marker=dict(color='teal'), name="Histogram"), row=1, col=1)
        hist_data = [self.dataframe[col].values, self.dataframe[col].values]
        group_labels = ['Group 1', 'Group 2']
        fig2 = ff.create_distplot(hist_data, group_labels)
        fig.add_trace(go.Scatter(fig2['data'][2],line=dict(color='blue', width=0.5), name="Distplot"), row=1, col=2)
        fig.update_layout(title="Distribution of Age", showlegend=False)
        return fig

    def plot_numerical_target(self,col,target):
        cat_li = []
        label_li=[]
        for cat in self.dataframe[target].unique():
            globals()[f"{target}{cat}"] = self.dataframe[self.dataframe[target] == cat][col]
            cat_li.append(eval(f"{target}{cat}"))
            label_li.append(f"{target}=={cat}")
        fig = ff.create_distplot(cat_li, group_labels=label_li, show_hist=False, show_rug=False)
        fig.update_layout(title=f"Distribution curves of {col} based on {target}")
        return fig

    def plot_ecdfplot(self,col):
        fig=px.ecdf(self.dataframe,x=col)
        fig.update_layout(title=f"Cumulative density plot of {col}")
        return fig

    def plot_boxplot(self,col):
        fig = px.box(self.dataframe, y=col)
        fig.update_layout(title=f"Box plot of {col}")
        return fig

    def plot_violinplot(self,col):
        fig = px.violin(self.dataframe, y=col)
        fig.update_layout(title=f"Violin plot of {col}")
        return fig



class Bivariate_viz:
    def __init__(self,dataframe):
        self.dataframe=dataframe

    def plot_countplot(self,cat_col1,cat_col2):
        data=self.dataframe.dropna(subset=[cat_col1])
        data=data.dropna(subset=[cat_col2])
        fig=px.histogram(data, x=cat_col1, color=cat_col2, barmode='group')
        fig.update_layout(title=f"Count plot of {cat_col1} distributed on {cat_col2}")
        return fig

    def plot_scatterplot(self,num_col1,num_col2,cat_col=None,reg_line=None):
        data=self.dataframe.dropna(subset=[num_col1])
        data=data.dropna(subset=[num_col2])
        if cat_col==None:
            pass
        else:
            data = data.dropna(subset=[cat_col])
        fig=px.scatter(data, x=num_col1, y=num_col2, color=cat_col, trendline=reg_line)
        if cat_col!=None:
            fig.update_layout(title=f"Scatter plot of {num_col1} and {num_col2} with {cat_col}")
        else:
            fig.update_layout(title=f"Scatter plot of {num_col1} and {num_col2}")
        return fig

    def plot_bi_boxplot(self,num_col,cat_col):
        data=self.dataframe.dropna(subset=[num_col])
        data=data.dropna(subset=[cat_col])
        fig = px.box(data, x=cat_col,y=num_col,color_discrete_sequence=['palevioletred'])
        fig.update_layout(title=f"Box plot of {num_col} distributed by {cat_col}")
        return fig

    def plot_bi_violinplot(self,num_col,cat_col):
        data=self.dataframe.dropna(subset=[num_col])
        data=data.dropna(subset=[cat_col])
        fig = px.violin(data, x=cat_col, y=num_col,color_discrete_sequence=['darkcyan'])
        fig.update_layout(title=f"Violin plot of {num_col} distributed by {cat_col}")
        return fig



class Multivariate_viz:

    def __init__(self,dataframe):
        self.dataframe=dataframe

    def plot_corr_heatmap(self,method):
        df_corr = self.dataframe.corr(method=method)
        x = list(df_corr.columns)
        y = list(df_corr.index)
        z = np.array(df_corr)
        fig = ff.create_annotated_heatmap(z,x=x,y=y,annotation_text=np.around(z, decimals=2),hoverinfo='x,y,z',colorscale='dense')
        fig.update_layout(title=f"Heatmap of {method.capitalize()} correlation")
        return fig

    def plot_corr_heatmap_target(self,target,method):
        df_corr = self.dataframe.corr(method=method)[target].sort_values(ascending=True)[:-1]
        x = [target]
        y = list(df_corr.index)
        z = np.array(df_corr).reshape(df_corr.shape[0], 1)
        fig = ff.create_annotated_heatmap(z,x=x,y=y,annotation_text=np.around(z, decimals=2),hoverinfo='y,z',colorscale='dense')
        fig.update_layout(title=f"{method.capitalize()} correlation heatmap on {target} column",autosize=False,width=500,height=600)
        return fig





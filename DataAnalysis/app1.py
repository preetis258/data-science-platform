# Importing the necessary libraries

import pandas as pd
import streamlit as st
from DataAnalysis.basic_info_script import BasicInformation
from DataAnalysis.target_info_script import TargetInformation
from DataAnalysis.features_info_script import ColumnInformation
from DataAnalysis.exploratory_data_analysis_script import Univariate_viz, Bivariate_viz,Multivariate_viz
import warnings
warnings.filterwarnings('ignore')

def show(i):
    if i==1:
        st.header( "Data Analysis and Model Building Tool" )
        st.subheader( "Upload a dataset, select a task in the left navigation bar and get things done!" )
    elif i==0:
        pass

######## page 1
def app_data_analysis(df):

    # title of the page
    st.title("Data Analysis Tool")

    ## Section for basic information about the dataset

    st.header("1. Data description")
    bi = BasicInformation(df)
    rows, cols = bi.dataset_information()
    st.markdown("##### Dataset ")
    shape_df = pd.DataFrame({'Rows': [rows], 'Columns': [cols]},index=None)
    hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """
    # Inject CSS with Markdown
    st.dataframe(df)
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.markdown("##### Number of rows and columns")
    st.table(shape_df)
    st.markdown("##### Columns and data types")
    dtype = df.dtypes.values.tolist()
    col = df.columns.tolist()
    st.table(pd.DataFrame(zip(col,dtype), columns=['Columns', 'Data type']).astype(str))

    # initializing the target column
    st.markdown("#### Select the target column")
    target_col=st.selectbox("",list(df.columns))
    st.markdown(f'Target column: {target_col}')

    # initializing ML task
    global ml_task
    task_pred=bi.target_column_checker(target_col)
    if task_pred=='classification':
        st.markdown("##### Select Machine Learning task")
        ml_task=st.radio("",['classification','regression'],index=0)
    else:
        st.markdown("##### Select Machine Learning task")
        ml_task = st.radio("", ['classification', 'regression'], index=1)
    st.markdown(f"#### Selected task: {ml_task} ")


##  Section for target column information
    st.header("2. Target description")
    ti=TargetInformation(df)
    # displaying category distribution
    if ml_task=='classification':
        try:
            unique_categories, class_size=ti.target_classification(target_col)
            target_df=pd.DataFrame({"Categories":unique_categories,"Counts":class_size})
            prct, imbalanced = ti.imbalance_checker(target_col)
            target_df['Percentages']=prct
            st.markdown("##### Category counts")
            st.table(target_df)

             # imbalance

            if imbalanced:
                st.markdown(f"##### Since the classes are not equally distributed hence {target_col} feature is imbalanced")
            else:
                st.markdown(f"##### Since the classes are almost equally distributed hence {target_col} feature is balanced")

            # countplot
            st.plotly_chart(ti.plot_countplot_target(target_col))
        except:
            st.markdown("Select appropriate options")

    else:
        try:
            st.markdown(f"##### Statistical measures of {target_col} column")
            st.table(ti.target_regression(target_col))
            st.markdown(f"##### Select visualization for {target_col} column")
            kde_input=st.radio(f"",['Histogram without distribution curve','Histogram with distribution curve','Only distribution curve'],index=0)
            if kde_input=='Histogram without distribution curve':
                st.plotly_chart(ti.plot_histplot_target(target_col,False,True))
            elif kde_input=='Histogram with distribution curve':
                st.plotly_chart(ti.plot_histplot_target(target_col,True,True))
            else:
                st.plotly_chart(ti.plot_histplot_target(target_col,True,False))

        except:
            st.markdown("Select appropriate options")

## Section for features information
    st.header("3. Column Description")
# #statistical description

    st.markdown("#### Missing values")
    ci=ColumnInformation(df)

## missing values
    st.markdown(f"Number of features having missing values: {ci.missing_value_features()[0]} ")
    st.markdown(f"##### Features with Missing values")
    if ci.missing_value_features()[0]!=0:
        st.table(ci.missing_value_features()[1])
    else:
        pass

## Outliers
    st.markdown("#### Outliers")
    outlier_det_sel=st.radio("Select outlier detection method: ",['Inter Quartile Range (IQR)','Percentile'],0)
    if outlier_det_sel=='Percentile':
        min_val,max_val=st.slider('Select lower and upper range values',1,100,(1, 100))
        outlier_df=ci.outier_features(outlier_det_sel,min_val,max_val)[1]
        if outlier_df.shape[0]!=0:
            st.table(outlier_df)
        else:
            pass
    elif outlier_det_sel=='Inter Quartile Range (IQR)':
        outlier_df = ci.outier_features(outlier_det_sel)[1]
        if outlier_df.shape[0]!=0:
            st.table(outlier_df)

## Duplicate values
    st.markdown("#### Duplicate rows")
    st.markdown(f"###### Number of duplicated values: {ci.number_of_duplicates_records()}")

## Section for exploratory data analysis
# univariate analysis
    st.header("4. Visualizations")
    st.subheader("Univariate plots")
    ua=Univariate_viz(df)
    cat_cols_more_10 = ua.features_segregation()[2]
    cat_cols = ua.features_segregation()[0]
    # categorical features

    if len(cat_cols)!=0:
        st.markdown("##### Categorical features")
        sel_cat_col=st.selectbox("Select a categorical feature:",cat_cols,index=0)
        st.plotly_chart(ua.plot_categorical(sel_cat_col))


        # if ml_task=='classification':
        #     st.plotly_chart(ua.plot_categorical_target(sel_cat_col,target_col))


    # numerical features
    num_cols = ua.features_segregation()[3]
    cat_cols_10 = ua.features_segregation()[1]
    if len(num_cols)!=0:
        st.markdown("##### Numerical features")
        sel_num_col = st.selectbox("Select a numerical feature:", num_cols )
        st.plotly_chart(ua.plot_numerical(sel_num_col))
        st.markdown("##### Statistical visualization")
        num_plot = st.radio("Select the plot",['Cumulative Density Function plot', 'Box plot', 'Violin plot'])
        num_col_univariate = st.selectbox("Select a column",num_cols)
        if num_plot=='Cumulative Density Function plot':
            st.plotly_chart(ua.plot_ecdfplot(num_col_univariate))
        elif num_plot=='Box plot':
            st.plotly_chart(ua.plot_boxplot(num_col_univariate))
        elif num_plot=='Violin plot':
            st.plotly_chart(ua.plot_violinplot(num_col_univariate))
    else:
        pass

# bivariate analysis
    st.subheader("Bivariate plots")
    ba = Bivariate_viz(df)

    # countplot
    if len(cat_cols_10)!=0:
        st.markdown("##### Count plot")
        first_box,second_box=st.columns(2)
        cat_cols_countplot=cat_cols_10.copy()
        sel_cat_col1=first_box.selectbox("1st feature:",cat_cols_countplot)
        sel_cat_col2=second_box.selectbox('2nd feature:',cat_cols_countplot)
        st.plotly_chart(ba.plot_countplot(sel_cat_col1,sel_cat_col2))
    else:
        pass

    # scatterplot
    if len(num_cols)!=0:
        st.markdown("##### Scatter plot")
        first_box_scatter, second_box_scatter = st.columns(2)
        num_cols_scatterplot = num_cols.copy()
        sel_num_col1 = first_box_scatter.selectbox("1st feature:", num_cols_scatterplot)
        sel_num_col2 = second_box_scatter.selectbox('2nd feature:', num_cols_scatterplot)
        cat_cols_scatter=cat_cols_10.copy()
        cat_cols_scatter.insert(0,'None')
        sel_cat_col_scatter=st.selectbox("Click to include a hue column",cat_cols_scatter)
        if sel_cat_col_scatter=='None':
            reg_line = st.radio("Insert regression line?", ["Yes", 'No'],1)
            if reg_line=='Yes':
                st.plotly_chart(ba.plot_scatterplot(sel_num_col1, sel_num_col2,None,'ols'))
            else:
                st.plotly_chart(ba.plot_scatterplot(sel_num_col1, sel_num_col2, None, None))
        else:
            reg_line = st.radio("Insert regression line?", ["Yes", 'No'],1)
            if reg_line == 'Yes':
                st.plotly_chart(ba.plot_scatterplot(sel_num_col1, sel_num_col2, sel_cat_col_scatter, 'ols'))
            else:
                st.plotly_chart(ba.plot_scatterplot(sel_num_col1, sel_num_col2, sel_cat_col_scatter, None))
        if len(cat_cols_10)!=0:
           # boxplot bivariate
            st.markdown("##### Box plot")
            first_box_box, second_box_box = st.columns(2)
            sel_num_box=first_box_box.selectbox("Select a numerical column:",num_cols)
            sel_col_box=second_box_box.selectbox("Select a categorical column:",cat_cols_10)
            st.plotly_chart(ba.plot_bi_boxplot(sel_num_box,sel_col_box))

            # violin bivariate
            st.markdown("##### Violin plot")
            first_box_violin, second_box_violin = st.columns(2)
            sel_num_violin = first_box_violin.selectbox("Select numerical column:", num_cols)
            sel_col_violin = second_box_violin.selectbox("Select categorical column:", cat_cols_10)
            st.plotly_chart(ba.plot_bi_violinplot(sel_num_violin, sel_col_violin))
        else:
            pass

        # Multivariate
        st.subheader("Multivariate plots")
        ma=Multivariate_viz(df)

        # correlation heatmap
        st.markdown("#### Correlation heatmap")
        corr_method=st.radio("Select the correlation method:",['Pearson','Spearman','Kendall Tau'])
        corr_name=corr_method.split()[0]
        corr_name=corr_name.lower()
        st.plotly_chart(ma.plot_corr_heatmap(method=corr_name))

        if ml_task=='regression':
            if len(num_cols)>1:
                st.plotly_chart(ma.plot_corr_heatmap_target(target_col,method=corr_name))
        else:
            pass
    else:
        pass





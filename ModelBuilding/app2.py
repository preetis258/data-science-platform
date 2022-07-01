## Importing the libraries
import streamlit as st
import pandas as pd
from ModelBuilding.basic_info_script import BasicInformation
from ModelBuilding.preprocessing_script import DataPreprocessing
from ModelBuilding.model_building_script import ModelBuilding
from sklearn.model_selection import train_test_split


def app_model_building(df):
    # title of the page
    st.title("Model Automation")

    ## Basic information
    bi = BasicInformation(df)
    # defining the target col
    target_col = st.selectbox("Select the target column", list(df.columns),key="1")
    st.markdown(f'#### Target column: {target_col}')

    # checking the problem type
    task_pred = bi.target_column_checker(target_col)
    if task_pred == 'Classification':
        ml_task = st.radio("Select Machine Learning task", ['Classification', 'Regression'], index=0)
    else:
        ml_task = st.radio("Select Machine Learning task", ['Classification', 'Regression'], index=1)
    st.markdown(f"#### Selected task: {ml_task}")

    if df.shape[1] > 1:
        # columns to drop
        col_to_drop=st.multiselect('Select the features which would not be required while training the model', list(df.columns))
        df.drop(col_to_drop, axis=1, inplace=True)

    ## Preprocessing

        data_prep=DataPreprocessing(df, target_col)
        st.markdown( f"##### Features: {df.columns.tolist()}" )
        num, cat=data_prep.num_cat_feat()
        st.markdown(f'\nNumber of numerical features: {len(num)}')
        st.markdown(f'Number of categorical features: {len(cat)}')

        st.subheader("Preprocessing")
        st.markdown("##### Missing value treatment")
        # missing value treatment
        if len(num)!=0:
            numerical_missing=st.radio('Select a missing value treament method for numerical features',
                                   ['Mean', 'Median', 'Zero'],index=0)
        else:
            numerical_missing=None

        categorical_missing = st.radio('Select a missing value treament method for categorical features',
                                       ['Mode', 'Unknown'],index=0)

        df = data_prep.missing_value_treatment(num_method=numerical_missing,cat_method=categorical_missing)[0]

        # outlier treatment
        if len(num)!=0:
            st.markdown("##### Outliers treatment")
            outlier_treatment = st.radio('Outliers treatment needed?', ['Yes','No'])
            if outlier_treatment == 'Yes':
                outlier_treatment_method = st.radio('Select a Outlier treament method ',['IQR','Percentile'])
                df = data_prep.outlier_treatment(method=outlier_treatment_method)
        else:
            pass

        # Dependent and Independent features encoding
        X, y = data_prep.independent_dependent_features()
            # Independent features encoding
        if len(cat)!=0:
            st.markdown("##### Categorical features encoding")
            independent_encoding_method = st.selectbox('Select a method for encoding categorical features into numerical',
                                                       ['OneHot', 'Ordinal'],key="2",index=1)
            X_encoded = data_prep.independent_feature_encoding(X,method=independent_encoding_method)[0]
            encoded_feats = data_prep.independent_feature_encoding(X,method=independent_encoding_method)[1]

        else:
            pass

        # dependent feature encoding
        if task_pred == 'Classification':
            y_encoded=data_prep.dependent_feature_encoding(y)
        else:
            y_encoded=y


    ## Model Building
        model_build=ModelBuilding(y_encoded)

        #scaling the independent features
        st.markdown("##### Feature scaling")
        whether_to_scale = st.radio('Scaling needed?', ['Yes', 'No'],index=1)
        if whether_to_scale == 'Yes':
            scaling_method=st.selectbox('Select a method for scaling:',['Min_Max','Standard','Robust'],key='3')
            X_scaled = model_build.feature_scaling(X=X_encoded, encoded_col=encoded_feats, method=scaling_method)
        else:
            X_scaled = X_encoded

        st.markdown("##### Top rows of the dataset after preprocessing")
        st.dataframe(X_scaled.head())

        # train test splitting
        st.subheader("Model Building")
        st.markdown("##### Splitting dataset for training and testing")
        test_ratio__ = st.slider('Select a test ratio', 0.01, 0.30)



        # Splitting dataset into training and testing sets
        def train_test_split_(X,y, problem_type, test_ratio_):
            if problem_type == 'Classification':
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_ratio_,
                                                                    random_state=0)
                return X_train, X_test, y_train, y_test

            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio_, random_state=0)

                return X_train, X_test, y_train, y_test

        X_train_, X_test_, y_train_, y_test_ = train_test_split_(X=X_scaled, y=y_encoded, problem_type=task_pred,
                                                                 test_ratio_ = test_ratio__)

        st.markdown(f'Training dataset shape: {X_train_.shape}')
        st.markdown(f'Testing dataset shape: {X_test_.shape}')

        # Model options
        if ml_task == 'Regression':
            model_name = st.multiselect('Select the model:',['Linear Regression','Support Vector Regressor','K-Nearest Neighbor',
                                                             'Decision Tree Regressor',
                                                             'Random Forest Regressor',
                                                             "AdaBoost Regressor",
                                                             'Gradient Boosting Regressor',
                                                             'XG Boost Regressor'])

        elif ml_task=='Classification':
            model_name = st.multiselect('Select the model:',['Logistic Regression',
                                                             "Support Vector Classifier",
                                                             "K-Nearest Neighbor",
                                                             "Decision Tree Classifier",
                                                             "Random Forest Classifier",
                                                             "AdaBoost Classifier",
                                                             "Gradient Boosting Classifier",
                                                             "XG Boost Classifier"])

        # Model training and results
        st.markdown("#### Selected models:")
        class_no=pd.Series(y_train_).nunique()
        if len(model_name)!=0:
            for idx,model in enumerate(model_name):
                st.markdown(f"{idx + 1}: {str(model)}")
            st.markdown("#### Select an evaluation metric")
            if ml_task == 'Regression':
                sel_metrics = st.radio("", ['r2 score', 'MSE', 'RMSE', 'MAE', 'MAPE'],
                                           index=0)
            elif ml_task == 'Classification':
                if class_no==2:
                    sel_metrics = st.radio(" ",
                                               ['Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score','AUC ROC score'], index=0)
                else:
                    sel_metrics = st.radio(" ",['Accuracy Score'])
            metrics_dict={}
            for idx,model in enumerate(model_name):
                fitted_model=model_build.model_training(X_train=X_train_,y_train=y_train_,X_test=X_test_,model_name =model,problem_type=ml_task)
                training_metrics=model_build.result_metrics(ml_task,y_train_,y_test_,fitted_model[0],fitted_model[1],class_prob=class_no)[0]
                testing_metrics=model_build.result_metrics(ml_task, y_train_, y_test_, fitted_model[0], fitted_model[1],class_prob=class_no)[1]
                metrics_dict[model]=model_build.results(training_metrics,testing_metrics,sel_metrics)
            df_metrics=pd.DataFrame(data=metrics_dict,index=['Training set','Testing set'])
            if st.button("Show results"):
                st.markdown(f"##### {sel_metrics}")
                st.table(df_metrics)


    else:
        st.markdown("#### Please a dataset with atleast one independent variable!")

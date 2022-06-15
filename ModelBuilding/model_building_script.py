#models
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,AdaBoostClassifier,AdaBoostRegressor,GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
import pandas as pd
#performance metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score, roc_auc_score


class ModelBuilding:

    def __init__(self,y):
        self.y = y


    def feature_scaling(self, X, encoded_col, method='Min_Max'):
        encoded_col_df = X[encoded_col]
        X.drop(columns=encoded_col,inplace=True)

        if method == 'Min_Max':
            # define min max scaler
            scaler = MinMaxScaler()
            # transform data
            X_scaled = scaler.fit_transform(X)
        elif method == 'Standard':
            # define standard scaler
            scaler = StandardScaler()
            # transform data
            X_scaled = scaler.fit_transform(X)
        elif method == 'Robust':
            # define standard scaler
            scaler = RobustScaler()
            # transform data
            X_scaled = scaler.fit_transform(X)

        X_scaled = pd.concat([pd.DataFrame(X_scaled), encoded_col_df], axis=1)

        return X_scaled

    r2_score_df1= pd.DataFrame(columns=['Model_Name','Train','Test'])

    def model_training(self,problem_type,model_name,X_train,y_train,X_test):

        reg_model_dict = {'Linear Regression': LinearRegression(),
                          'Support Vector Regressor': SVR(),
                          'K-Nearest Neighbor': KNeighborsRegressor(),
                          'Decision Tree Regressor': DecisionTreeRegressor(),
                          'Random Forest Regressor': RandomForestRegressor(),
                          "AdaBoost Regressor":AdaBoostRegressor(),
                          'Gradient Boosting Regressor': GradientBoostingRegressor(),
                          'XG Boost Regressor': XGBRegressor()}

        class_model_dict={'Logistic Regression':LogisticRegression(),
                          "Support Vector Classifier":SVC(),
                          "K-Nearest Neighbor":KNeighborsClassifier(),
                          "Decision Tree Classifier":DecisionTreeClassifier(),
                          "Random Forest Classifier":RandomForestClassifier(),
                          "AdaBoost Classifier":AdaBoostClassifier(),
                          "Gradient Boosting Classifier":GradientBoostingClassifier(),
                          "XG Boost Classifier":XGBClassifier()}

        if problem_type == 'Regression':
            model=reg_model_dict[model_name]
        elif problem_type == 'Classification':
            model = class_model_dict[model_name]

        model.fit(X_train, y_train)
        train_pred = model.predict( X_train )
        test_pred = model.predict( X_test )
        return train_pred,test_pred

    def result_metrics(self,problem_type,y_train_true,y_test_true,y_train_pred,y_test_pred):

        def reg_metrics(y_true,y_pred):
            reg_metrics_dict={"r2 score":r2_score(y_true,y_pred),
                           "MSE":mean_squared_error(y_true,y_pred),
                           "RMSE":mean_squared_error(y_true,y_pred)**(1/2),
                           "MAE":mean_absolute_error(y_true,y_pred),
                           "MAPE":mean_absolute_percentage_error(y_true,y_pred)}
            return reg_metrics_dict

        def class_metrics(y_true,y_pred):
            class_metrics_dict={"Accuracy Score":accuracy_score(y_true,y_pred),
                                "Precision Score":precision_score(y_true,y_pred),
                                "Recall Score":recall_score(y_true,y_pred),
                                "F1 Score":f1_score(y_true,y_pred),
                                "AUC ROC score":roc_auc_score(y_true,y_pred)}
            return class_metrics_dict

        if problem_type=='Regression':
            training_metrics=reg_metrics(y_train_true,y_train_pred)
            testing_metrics=reg_metrics(y_test_true,y_test_pred)

        elif problem_type=='Classification':
            training_metrics = class_metrics( y_train_true, y_train_pred )
            testing_metrics = class_metrics( y_test_true, y_test_pred )

        return training_metrics,testing_metrics

    def results(self,training_metrics_dic,testing_metrics_dic,metrics):
        metrics_li=[]
        metrics_li.append(training_metrics_dic[metrics])
        metrics_li.append(testing_metrics_dic[metrics])
        return metrics_li









        # st.text(training_metrics)
        # st.text(testing_metrics)
        # st.dataframe(pd.DataFrame(training_metrics))



            # for key,sets in predictions.items():
            #     globals()[f"{key}_metrics"] = reg_metrics(sets[0],sets[1])
            # training_metrics




        #
        #     performance_met = []
        #     train_res = []
        #     test_res = []
        #
        #     # r2
        #     performance_met.append('R2')
        #     train_r2_score=r2_score(y_train, train_pred)
        #     train_res.append(round(train_r2_score,2))
        #     test_r2_score=r2_score(y_test, test_pred)
        #     test_res.append(round(test_r2_score,2))
        #
        #     #mean_squared_error
        #     performance_met.append('Mean Square Error')
        #     train_mse_score = mean_squared_error(y_train, train_pred)
        #     train_res.append(round(train_mse_score,2))
        #     test_mse_score = mean_squared_error(y_test, test_pred)
        #     test_res.append(round(test_mse_score,2))
        #
        #     # Root_mean_squared_error
        #     performance_met.append('Root Mean Square Error')
        #     train_rmse_score = mean_squared_error(y_train, train_pred)
        #     train_res.append(round(np.sqrt(train_rmse_score),2))
        #     test_rmse_score = mean_squared_error(y_test, test_pred)
        #     test_res.append(round(np.sqrt(test_rmse_score),2))
        #
        #     # mean_absolute_error
        #     performance_met.append('Mean Absolute Error')
        #     train_mse_score = mean_absolute_error(y_train, train_pred)
        #     train_res.append(round(train_mse_score,2))
        #     test_mse_score = mean_absolute_error(y_test, test_pred)
        #     test_res.append(round(test_mse_score,2))
        #
        #     # mean_absolute_percentage_error
        #     performance_met.append('Mean Absolute Percent Error')
        #     train_mse_score = mean_absolute_percentage_error(y_train, train_pred)
        #     train_res.append(round(train_mse_score,2))
        #     test_mse_score = mean_absolute_percentage_error(y_test, test_pred)
        #     test_res.append(round(test_mse_score,2))
        #
        #
        #     r2_score_df=pd.DataFrame(list(zip(performance_met, train_res, test_res)),
        #                              columns=['Evaluation_mterics', 'Train', 'Test'])
        #     # r2_df=r2_df.append(r2_score_df)
        #     return r2_score_df
        # #     st.table(r2_score_df)
        # st.table(r2_df)
      #
      # #classification
      #
      #   elif problem_type == 'Classification':
      #       # def regression_model(model_name):
      #       if model_name == 'Logistic':
      #           logistic_reg = LogisticRegression()
      #           logistic_reg.fit(X_train, y_train)
      #           train_pred=logistic_reg.predict(X_train)
      #           test_pred=logistic_reg.predict(X_test)
      #
      #
      #       elif model_name == 'SVM':
      #           svm_clas = SVC()
      #           svm_clas.fit(X_train, y_train)
      #           train_pred = svm_clas.predict(X_train)
      #           test_pred = svm_clas.predict(X_test)
      #
      #       elif model_name == 'KNN':
      #           knn_clas = KNeighborsClassifier(n_neighbors=5)
      #           knn_clas.fit(X_train,y_train)
      #           train_pred = knn_clas.predict(X_train)
      #           test_pred = knn_clas.predict(X_test)
      #
      #
      #       elif model_name == 'DecisionTree':
      #           decision_tree_clas = DecisionTreeClassifier(random_state=0)
      #           decision_tree_clas.fit(X_train, y_train)
      #           train_pred = decision_tree_clas.predict(X_train)
      #           test_pred = decision_tree_clas.predict(X_test)
      #
      #       elif model_name == 'RandomForest':
      #           random_for_clas = RandomForestClassifier(random_state=0)
      #           random_for_clas.fit(X_train, y_train)
      #           train_pred = random_for_clas.predict(X_train)
      #           test_pred = random_for_clas.predict(X_test)
      #
      #       elif model_name=='GradientBoosting':
      #           gb_clas=GradientBoostingClassifier()
      #           gb_clas.fit(X_train,y_train)
      #           train_pred = gb_clas.predict(X_train)
      #           test_pred = gb_clas.predict(X_test)
      #
      #
      #
      #       elif model_name=='XGBoosting':
      #           xgb_clas=XGBClassifier()
      #           xgb_clas.fit(X_train,y_train)
      #           train_pred = xgb_clas.predict(X_train)
      #           test_pred = xgb_clas.predict(X_test)
      #
      #
      #
      #       performance_met = []
      #       train_res = []
      #       test_res = []
      #
      #       # accuracy
      #       performance_met.append('Accuracy')
      #       train_r2_score=accuracy_score(y_train, train_pred)
      #       train_res.append(round(train_r2_score,2))
      #       test_r2_score=accuracy_score(y_test, test_pred)
      #       test_res.append(round(test_r2_score,2))
      #
      #       # f1
      #       performance_met.append('F1-Score')
      #       train_mse_score = f1_score(y_train, train_pred)
      #       train_res.append(round(train_mse_score,2))
      #       test_mse_score = f1_score(y_test, test_pred)
      #       test_res.append(round(test_mse_score,2))
      #
      #       # precision
      #       performance_met.append('Precision')
      #       train_rmse_score = precision_score(y_train, train_pred)
      #       train_res.append(round(train_rmse_score,2))
      #       test_rmse_score = precision_score(y_test, test_pred)
      #       test_res.append(round(test_rmse_score,2))
      #
      #       # recall
      #       performance_met.append('Recall')
      #       train_mse_score = recall_score(y_train, train_pred)
      #       train_res.append(round(train_mse_score,2))
      #       test_mse_score = recall_score(y_test, test_pred)
      #       test_res.append(round(test_mse_score,2))
      #
      #       # roc-auc
      #       performance_met.append('ROC-AUC')
      #       train_mse_score = roc_auc_score(y_train, train_pred)
      #       train_res.append(round(train_mse_score,2))
      #       test_mse_score = roc_auc_score(y_test, test_pred)
      #       test_res.append(round(test_mse_score,2))
      #
      #
      #       r2_score_df=pd.DataFrame(list(zip(performance_met, train_res, test_res)),
      #                                columns=['Evaluation_mterics', 'Train', 'Test'])
      #       # r2_df=r2_df.append(r2_score_df)
      #       return r2_score_df
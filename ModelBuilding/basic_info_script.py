class BasicInformation:
    def __init__(self, dataframe):
        self.dataframe = dataframe


    def target_column_checker(self, target_column):
        # this function will try to identify whether the problem is of regression or classification
        if (self.dataframe[target_column].dtype != 'O') & (self.dataframe[target_column].nunique()<=10):
            return 'Classification'
        elif self.dataframe[target_column].dtype == 'O':
            return 'Classification'

        elif (self.dataframe[target_column].dtype != 'O') & (self.dataframe[target_column].nunique()>10):
            return 'Regression'


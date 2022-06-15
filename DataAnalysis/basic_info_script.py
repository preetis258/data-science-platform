
class BasicInformation:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def dataset_information(self):
        '''a  function that takes dataframe as an argument and return the shape and size of the dataframe'''
        shape = self.dataframe.shape
        return shape[0],shape[1]

    def target_column_checker(self,target_column):
        '''checks whether the task is regression or classification'''
        if (self.dataframe[target_column].dtype=='int64') & (self.dataframe[target_column].nunique()<5):
            return 'classification'
        elif (self.dataframe[target_column].dtype=='float64') & (self.dataframe[target_column].nunique()<5):
            return 'classification'
        elif (self.dataframe[target_column].dtype=='O'):
            return 'classification'
        elif (self.dataframe[target_column].dtype!='O') & (self.dataframe[target_column].nunique()>5):
            return 'regression'
        else:
            return 'regression'



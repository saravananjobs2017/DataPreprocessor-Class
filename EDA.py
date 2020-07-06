import pandas as pd
import numpy as numpy
class DataPreprocessor:
    def __init__(self):
        pass
    def impute_missing_values(self, data,target=None, strategy='median', impute_val=None,missing_vals=None,mv_flag=None):
        """
                Method Name: impute_missing_values
                Description: This method will be used to impute missing values in the dataframe

                Input Description:
                data: Name of the input dataframe

                strategy: Strategy to be used for MVI (Missing Value Imputation)
                --‘median’ : default for continuous variables,
                 replaces missing value(s) with median of the concerned column
                --‘mean’
                --‘mode’ : default for categorical variables
                --‘fixed’ : replaces all missing values with a fixed ‘explicitly specified’ value

                impute_val: None(default), can be assigned a value to be used for imputation in ‘fixed’ strategy

                missing_vals: None(default), a list/tuple of missing value indicators. By default,
                 it considers only NaN as missing. Dictionary can be passed to consider different missing values
                for different columns in format – {col_name:[val1,val2, …], col2:[…]}

                mv_flag: None(default), can be passed list/tuple of columns as input for which it creates missing
                value flags

                return: A DataFrame with missing values imputed

                Written By: Purvansh singh
                Version: 1.0
                Revisions: None
                """
        if isinstance(data, pd.DataFrame) and not data.empty:
            dataframe = data
            # Converting missing_vals to Nan Values
            if mv_flag is True:
                dataframe.replace(missing_vals,np.nan,inplace=True)
            #  Checking for Missing Value in Dependent Variable
            if dataframe[target].isna().any():
                target_column = dataframe[target]
                dataframe.drop(target,axis=1,inplace=True)
                #dataframe[target].dropna(inplace=True)  # If any missing Value found Dropping those
            # Checking for Missing Values in Independent Variable
            Missing_data_columns = dataframe.columns[dataframe.isna().any()].tolist()  # Finding Columns with the missing data from dataframe
            if strategy == 'fixed': # checking if strategy == fixed
                dataframe.fillna(impute_val, inplace=True) # Filling the Nan values with the imputed value from user
            else:
                for columns in Missing_data_columns:  # Iterating over the columns having Nan Values
                    if dataframe[columns].dtype == 'object': # Checking for the categorical data
                        Mode = dataframe[columns].mode()[0]
                        dataframe[columns].fillna(Mode, inplace=True)  # Imputing Nan values with mode of the column
                    else:
                        if strategy == 'median': # checking if the strategy == median
                            Median = dataframe[columns].median()
                            dataframe[columns].fillna(Median, inplace=True)  # Imputing Nan values with median of the column                        else:  # The only strategy remains is mean
                            Mean = dataframe[columns].mean()
                            dataframe[columns].fillna(Mean, inplace=True)   # Imputing Nan values with mean of the column
            dataframe[target] = target_column
            dataframe[target].dropna(inplace=True)
        else:
            pass
        return dataframe

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    final = DataPreprocessor().impute_missing_values(df, 'fixed', 20)









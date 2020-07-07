import numpy as np
import pandas as pd


class DataPreprocessor:
    def __init__(self,logger_object):
        self.logger_object = logger_object

    def impute_missing_values(self, data, mv_flag=None, target=None, strategy='median', impute_val=None,
                              missing_vals=None):
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

            if mv_flag is True:
                # Converting missing_vals to Nan Values
                if missing_vals:
                    dataframe.replace(missing_vals, np.nan, inplace=True)
                #  Checking for Missing Values in Dependent Variable
                if dataframe[target].isna().any():
                    dataframe = dataframe[dataframe[
                        target].notna()].copy()  # Selecting the Dataframe With No missing values in Dependent column
                # Checking for Missing Values in Independent Variables
                Missing_data_columns = dataframe.columns[
                    dataframe.isna().any()].tolist()  # Finding Columns with the missing data from dataframe
                if strategy == 'fixed':  # checking if strategy == fixed
                    dataframe.fillna(impute_val,
                                     inplace=True)  # Filling the Nan values with the imputed value from user
                else:
                    for columns in Missing_data_columns:  # Iterating over the columns having Nan Values
                        if dataframe[columns].dtype == 'object':  # Checking for the categorical data
                            Mode = dataframe[columns].mode()[0]
                            dataframe[columns].fillna(Mode, inplace=True)  # Imputing Nan values with mode of the column
                        else:
                            if strategy == 'median':  # checking if the strategy == median
                                Median = dataframe[columns].median()
                                dataframe[columns].fillna(Median,
                                                          inplace=True)  # Imputing Nan values with median of the column
                            else:  # The only strategy remains is mean
                                Mean = dataframe[columns].mean()
                                dataframe[columns].fillna(Mean,
                                                          inplace=True)  # Imputing Nan values with mean of the column

            else:
                pass

        return dataframe

    def type_conversion(self,dataset, cat_to_num=None,num_to_cat=None):
        '''

            Method Name: type_conversion
            Description: This method will be used to convert column datatype from
            numerical to categorical or vice-versa, if possible.

            Input Description:

            dataset: input DataFrame in which type conversion is needed

            cat_to_num: None(default),list/tuple of variables that need to
            be converted from categorical to numerical

            num_to_cat: None(default),list/tuple of variables to be
            converted from numerical to categorical

            return: A DataFrame with column types changed as per requirement

            Written By: Purvansh singh
            Version: 1.0
            Revisions: None
        '''
        self.logger_object.log("Entered into type_conversion method.")
        if isinstance(dataset, pd.DataFrame) and not dataset.empty:
            if cat_to_num is not None:
                for column in cat_to_num:
                    try:
                        dataset[column] = pd.to_numeric(dataset[column])
                    except Exception as e:
                        print(e)

            if num_to_cat is not None:
                for column in num_to_cat:
                    try:
                        dataset[column] = dataset[column].astype('object')
                    except Exception as e:
                        print(e)

            return dataset




if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    print(df.shape)
    print(df.isna().sum())
    final = DataPreprocessor().impute_missing_values(df, True, 'is_promoted', 'fixed', 5000,
                                                     {'recruitment_channel': 'other'})

    print(final.head())
    print(final.isna().sum())
    print(final.shape)

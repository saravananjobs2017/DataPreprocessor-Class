import numpy as np
import pandas as pd
import logger
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    """
        This class shall be used to include all Data Preprocessing techniques to be feed to the Machine Learning Models

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def impute_missing_values(self, data, mv_flag=None, target=None, strategy='median', impute_val=None,
                              missing_vals=None):
        """
                Method Name: impute_missing_values
                Description: This method will be used to impute missing values in the dataframe

                Input Description:
                data: Name of the input dataframe

                target: Name of the target column of DataFrame

                strategy: Strategy to be used for MVI (Missing Value Imputation)
                --‘median’ : default for continuous variables,
                 replaces missing value(s) with median of the concerned column
                --‘mean’
                --‘mode’ : default for categorical variables
                --‘fixed’ : replaces all missing values with a fixed ‘explicitly specified’ value

                impute_val: None(default), can be assigned a value to be used for imputation i
                n ‘fixed’ strategy

                missing_vals: None(default), a list/tuple of missing value indicators. By default,
                 it considers only NaN as missing. Dictionary can be passed to consider different missing values
                for different columns in format – {col_name:[val1,val2, …], col2:[…]}

                mv_flag: None(default), can be passed list/tuple of columns as input for which it creates missing
                value flags

                On Exception: Write the exception in the log file. Raise an exception with the appropriate error message

                return: A DataFrame with missing values imputed

                Written By: Purvansh singh
                Version: 1.0
                Revisions: None
                """
        self.logger_object.log(self.file_object, "Entered into impute_missing_values method.")
        try:
            if isinstance(data, pd.DataFrame) and not data.empty:
                self.logger_object.log(self.file_object, "Non-empty DataFrame object Identified")
                dataframe = data

                if mv_flag is True:
                    self.logger_object.log(self.file_object, "my_flag found True Imputing Dataframe.")
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
                                dataframe[columns].fillna(Mode,
                                                          inplace=True)  # Imputing Nan values with mode of the column
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
                    self.logger_object.log(self.file_object, "my_flag found False")
            else:
                raise Exception("No DataFrame Found")
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   "Error Occurred in impute_missing_values, Error statement: " + str(e))
            raise Exception(e) from None  # Suppressing the Error Chaining
        else:
            self.logger_object.log(self.file_object, "Imputed DataFrame Returned Successfully")
            return dataframe

    def type_conversion(self, dataset, cat_to_num=None, num_to_cat=None):
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

            On Exception : Write the exception in the log file. Raise an exception with the appropriate error message

            Written By: Purvansh singh
            Version: 1.0
            Revisions: None
        '''
        self.logger_object.log(self.file_object, "Entered into type_conversion method.")
        try:
            if isinstance(dataset, pd.DataFrame) and not dataset.empty:
                self.logger_object.log(self.file_object, "Non-empty DataFrame object Identified")
                if cat_to_num is not None:
                    for column in cat_to_num:
                        dataset[column] = pd.to_numeric(dataset[column])

                if num_to_cat is not None:
                    for column in num_to_cat:
                        dataset[column] = dataset[column].astype('object')
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   "Error Occurred in type_conversion method, Error statement: " + str(e))
            raise Exception(e) from None  # Suppressing the Error Chaining
        else:
            self.logger_object.log(self.file_object, "type_converted DataFrame Returned Successfully")
            return dataset

    def pca(self, data, var_explained):
        """
        Method Name: pca
        Description: This method reduces the dimension from scaled Data which enables
                     quick for large data files.

        input      : Data which is Scaled since PCA works on continous data
                     var_explained is no_components in PCA , default value passed.

        Output     : It returns the scaled and reduced dimensions.

        On Failure : Raise Exception

        Written by : iNeuron Intelligence

        version    : 1.0

        revisions  : None.


        """

        self.data = data
        self.var_explained = var_explained
        pca = PCA()
        self.logger_object.log(self.file_object, 'S::Entered the PCA method of the DataPreprocessor class')
        try:

            # initializing with different combination of parameters
            self.parameter_grid = {"n_components": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   "svd_solver": ['auto', 'full', 'arpack', 'randomized']}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=PCA, param_grid=parameter_grid, cv=3, verbose=3)

            # Finding Best parameters
            grid.fit(data)

            # extracting the best parameters
            self.n_components = grid.best_params_['n_components']
            self.svd_solver = grid.best_params_['svd_solver']

            # creating pca dataset with the best parameters
            pca = PCA(n_components=self.n_components, svd_solver=self.svd_solver)
            # applying PCA model
            pca.fit(data)
            Exp_var = var_explained * 10  ##pca.explained_variance_ratio_#default=0.90
            self.logger_object.log(self.file_object, 'I : PCA best params: ' + str(
                self.grid.best_params_))

            x_pca = pca.transform(data)
            # Make Dataframe with pca data
            columns = ['PC' + str(i) for i in range(1, Exp_var + 1)]
            pca_data = pd.DataFrame(data=x_pca, columns=columns)

            self.logger_object.log(self.file_object, "C:: the PCA method of the DataPreprocessor class')
            return pca_data


        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'E : Exception occured in PCA method of the Data Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'E : Unsuccessful. Exited the PCA method of the DataPreprocessor class')

            raise Exception()

    def remove_imbalance(self, data, target, threshold=10.0, oversample=True, smote=False):
        """
        Method Name: remove_imbalance
        Description: This method will be used to handle unbalanced datasets(rare classes) through oversampling/ undersampling 
                     techniques
        Input Description: data: the input dataframe with target column.
                           threshold: the threshold of mismatch between the target values to perform balancing.

        Output: A balanced dataframe.
        On Failure: Raise Exception

        Written By: Punit Nanda
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object,'Entered the remove_imbalance method of the DataPreprocessor class') # Logging entry to the method
        try:
            #data= pd.read_csv(self.training_file) # reading the data file
            self.logger_object.log(self.file_object,'DataFrame Load Successful of the remove_imbalance method of the DataPreprocessor class')
            #return self.data # return the read data to the calling method
            
            self.logger_object.log(self.file_object,'X y created in the remove_imbalance method of the DataPreprocessor class')
            X = data.drop(target, axis=1)
            y = data[target]
            
            self.logger_object.log(self.file_object,'Class Imbalance Process Starts in the remove_imbalance method of the DataPreprocessor class')
            
            no_of_classes = data[target].nunique()
            
            if no_of_classes == 2:
                
                self.logger_object.log(self.file_object,'No of Classes is 2 in the remove_imbalance method of the DataPreprocessor class')
                thresh_satisfied = ((data[target].value_counts()/float(len(data[target]))*100).any() < threshold)
                if thresh_satisfied:
                    self.logger_object.log(self.file_object,'Threshold satisfied in the remove_imbalance method of the DataPreprocessor class')
                    if smote:
                        self.logger_object.log(self.file_object,'OverSampling using SMOTE having 2 classes in the remove_imbalance method of the DataPreprocessor class')
                        smote = SMOTE()
                        X, y = smote.fit_resample(X, y)
                    elif oversample:
                        self.logger_object.log(self.file_object,'OverSampling minority classes data having 2 classes in the remove_imbalance method of the DataPreprocessor class')
                        ROS = RandomOverSampler(sampling_strategy='auto', random_state=42)
                        X, y = ROS.fit_sample(X, y)
                    else:
                        self.logger_object.log(self.file_object,'UnderSampling majority classes data having 2 classes in the remove_imbalance method of the DataPreprocessor class')
                        ROS = RandomUnderSampler(sampling_strategy='auto', random_state=42)
                        X, y = ROS.fit_sample(X, y)
            else:
                
                high = (data[target].value_counts()/float(len(data[target]))*100).ravel().max()
                low = (data[target].value_counts()/float(len(data[target]))*100).ravel().min()
                
                thresh_satisfied = ( high-low > 100.0 - threshold )
                
                if thresh_satisfied:
                    self.logger_object.log(self.file_object,'Threshold satisfied in the remove_imbalance method of the DataPreprocessor class')
                    if smote:
                        self.logger_object.log(self.file_object,'OverSampling using SMOTE having more than 2 classes in the remove_imbalance method of the DataPreprocessor class')
                        for i in range(no_of_classes-2):
                            smote = SMOTE()
                            X, y = smote.fit_resample(X, y)
                    elif oversample:
                        self.logger_object.log(self.file_object,'OverSampling minority classes data having more than 2 classes in the remove_imbalance method of the DataPreprocessor class')
                        for i in range(no_of_classes-2):
                            ROS = RandomOverSampler(sampling_strategy='auto', random_state=42)
                            X, y = ROS.fit_sample(X, y)
                    else:
                        self.logger_object.log(self.file_object,'UnderSampling majority classes data having more than 2 classes in the remove_imbalance method of the DataPreprocessor class')
                        for i in range(no_of_classes-2):
                            ROS = RandomUnderSampler(sampling_strategy='auto', random_state=42)
                            X, y = ROS.fit_sample(X, y)                    
                                                     
            
            y.to_frame(name=target)
            dfBalanced = pd.concat([X, y], axis=1)
            self.logger_object.log(self.file_object,'Class Imbalance Process Ends in the remove_imbalance method of the DataPreprocessor class')
            return dfBalanced
            
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_imbalance method of the DataPreprocessor class. Exception message: '+str(e)) # Logging the exception message
            self.logger_object.log(self.file_object,
                                   'DataFrame Load Unsuccessful.Exited the remove_imbalance method of the DataPreprocessor class') # Logging unsuccessful load of data
            raise Exception() # raising exception and exiting

    def remove_columns_with_minimal_variance(self, data, threshold):
        """
        Method Name: remove_columns_with_minimal_variance
        Description: This method drops any numerical column with standard deviation below specified threshold
        Input Parameter Description: data: input DataFrame in which we need to check std deviations
                                     threshold : the threshold for std deviation below which we need to drop the columns


        Output: A DataFrame with numerical columns with low std dev dropped.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object,
                               'Entered the remove_columns_with_minimal_variance method of the DataPreprocessor class')  # Logging entry to the method
        try:
            # self.logger_object.log(self.file_object,'Data Load Successful.') # Logging exit from the method
            sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
            sel_var = sel.fit_transform(data)
            new_data = data[data.columns[sel.get_support(indices=True)]]
            return new_data  # return the read data to the calling method
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in remove_columns_with_minimal_variance method of the DataPreprocessor class. Exception message: ' + str(
                                       e))  # Logging the exception message
            raise Exception()  # raising exception and exiting

    def standardize_data(self, dataframe):

        """
        Method Name: standardize_data
        Description: This method will be used to standardize al the numeric variables. Where mean = 0, std dev = 1.
        Input Description: data: the input dataframe with numeric columns.

        Output: Standardized data where mean of each column will be 0 and standard deviation will be 1.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            data = dataframe
            stdscalar = StandardScaler()
            scaled_data = stdscalar.fit_transform(data)
            return scaled_data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured while stadardizing data. Exception message:  ' + str(e))
            raise Exception()

    def normalize_data(self, dataframe):

        """
        Method Name: normalize_data
        Description: This method will be used to normalize all the numeric variables. Where min value = 0 and max value = 1.
        Input Description: data: the input dataframe with numeric columns.

        Output: Normalized data where minimum value of each column will be 0 and maximum value of each column will be 1.
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        try:
            data = dataframe
            normalizer = MinMaxScaler()
            normalized_data = normalizer.fit_transform(data)
            return normalized_data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured while stadardizing data. Exception message:  ' + str(e))
            raise Exception()

if __name__ == '__main__':
    log = logger.App_Logger()
    df = pd.read_csv('train.csv')
    file = open('log.txt', "a")
    print(df.dtypes)
    # Test your code by calling methods here.



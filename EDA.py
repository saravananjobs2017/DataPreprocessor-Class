class DataPreprocessor:
    def __init__(self):
        pass
    def impute_missing_values(self, data, strategy = None, impute_val = None , missing_vals, mv_flag):
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
            Numerical_columns = data.select_dtypes(include=[np.number])  # Selecting Numerical Columns from Dataset
            Categorical_columns = data.select_dtypes(include=[object])  # Selecting Categorical Columns from Dataset
            if strategy is None: # When no strategy argument is passed Default conditions
                Numerical_Nan_columns = Numerical_columns.columns[Numerical_columns.isna().any()].tolist() # Selecting the columns having null value
                for columns in Numerical_Nan_columns: # Iterating through all the columns having null values and imputing them.
                     data[columns].fillna(data[columns].median(),inplace=True)
                Categorical_Nan_columns = Categorical_columns.columns[Categorical_columns.isna().any()].tolist() # Selecting the columns having null value
                for columns in Categorical_Nan_columns:

                        # Iterating through all the columns having null values and imputing them.










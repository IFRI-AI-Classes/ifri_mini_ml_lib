import numpy as np
import pandas as pd


class Univariate_Treatment:
    """
    This class handles the treatment of outliers in univariate data.
    It receives a dataframe and the outliers detected by the detection step,
    then applies the chosen treatment method column by column.
    
    """
    def __init__(self, dataframe):
        """ We never work on the original dataset because any modification
        to the class will modify the user's original dataframe."""
        self.dataframe = dataframe.copy()
        
        
    def _get_numeric_columns(self):
        """We select only the numeric columns of the dataframe."""
        return self.dataframe.select_dtypes(include=np.number).columns
    
    
    
    #------First Method: Deletion------
    
    def remove_outliers(self, outlier_indices):
        """
        Removes the rows identified as outliers from the dataframe.
        Best used when the dataset is large enough that losing rows is acceptable.

        Args:
            outlier_indices: list or array of row indices to remove

        Returns:
            DataFrame with outlier rows dropped
        """
        
        #If index is in  outlier_indices , we drp the row
        cleaned_df = self.dataframe.drop(index=outlier_indices, errors="ignore")
        print(f"{len(outlier_indices)} outlier(s) removed. "
            f"Rows before: {len(self.dataframe)} → after: {len(cleaned_df)}")

        return cleaned_df
    
    
    #------Second Method--------
    def winsorize(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        Winsorize principle:
        We choose percentiles:

        -low: e.g., 5%
        -high: e.g., 95%
        If x < lower quantile, then x is replaced by the lower quantile.
        If x > upper quantile, then x is replaced by the upper quantile.
        This keeps all rows but limits extreme values.

        Args:
            lower_quantile (float): lower bound quantile. Defaults to 0.05
            upper_quantile (float): upper bound quantile. Defaults to 0.95

        Returns:
            DataFrame with capped values
        """ 
        
        df_capped = self.dataframe.copy()

        for column in self._get_numeric_columns():
            # Compute the lower and upper bounds for this column
            lower_bound = df_capped[column].quantile(lower_quantile)
            upper_bound = df_capped[column].quantile(upper_quantile)

            # clip() is a pandas method whose replaces values below lower_bound with lower_bound
            # and values above upper_bound with upper_bound
            df_capped[column] = df_capped[column].clip(
                lower=lower_bound,
                upper=upper_bound
            )

            print(f"Column '{column}' → capped to [{lower_bound:.2f}, {upper_bound:.2f}]")

        return df_capped
        
    #------Third Method--------
    
    def impute_median(self, outlier_indices):
        """
        Replaces outlier values with the median of the column,
        computed only on the non-outlier points (inliers).
        The median is robust to skewed distributions.

        Args:
            outlier_indices: list or array of row indices identified as outliers

        Returns:
            DataFrame with outlier values replaced by the column median
        """
        df_imputed = self.dataframe.copy()
        
        inlier_mask = ~df_imputed.index.isin(outlier_indices)
        """
        The `inlier` will return a boolean array where outliers will have the value `true` and
        normal points will have the value `false`. The `~` symbol will therefore invert the results, 
        giving `true` for normal points and `false` for outliers.
        """
        for column in self._get_numeric_columns():
            # Compute the median using only the inlier rows
            median_value = df_imputed.loc[inlier_mask, column].median()

            # Replace outlier values in this column with the median
            df_imputed.loc[outlier_indices, column] = median_value
            print(f"Column '{column}' → outliers replaced by median = {median_value:.2f}")

        return df_imputed
    
    
    #------Fourth Method--------
    def impute_mean(self, outlier_indices):
        """
        Replaces outlier values with the mean of the column,
        computed only on the inlier points.
        Best used when the distribution is approximately normal.

        Args:
            outlier_indices: list or array of row indices identified as outliers

        Returns:
            DataFrame with outlier values replaced by the column mean
        """
        df_imputed = self.dataframe.copy()

        # Inlier rows are all rows that are not outliers
        inlier_mask = ~df_imputed.index.isin(outlier_indices)

        for column in self._get_numeric_columns():
            # Compute the mean using only the inlier rows
            mean_value = df_imputed.loc[inlier_mask, column].mean()

            # Replace outlier values with the mean
            df_imputed[column] = df_imputed[column].astype(float)
            df_imputed.loc[outlier_indices, column] = mean_value

            print(f"Column '{column}' outliers replaced by mean = {mean_value:.2f}")

        return df_imputed
    
    
    #------Five Method--------
    def log_transform (self, columns=None):
     """ Applies a log(1 + x) transformation to reduce the effect of extreme values.
        This compresses large values without removing them.
        Only applicable to non-negative columns.
        log(1 + x) instead of log(x) to handle zeros safely (log(0) = -inf)
        ex:
        before log:
        [10, 12, 15, 20, 5000]
        after log:
        [2.3, 2.4, 2.7, 3.0, 8.5]
        
        Args:
            columns: list of column names to transform.
                     If None, applies to all numeric columns.

        Returns:
            DataFrame with log-transformed columns
     """
     
     df_log = self.dataframe.copy()
     
     target_columns = columns if columns else self._get_numeric_columns()
     for column in target_columns:
            # Check that all values are non-negative 
            if (df_log[column] < 0).any():
                print(f"Column '{column}' skipped : contains negative values "
                      f"(log transform requires non-negative values)")
                continue
            
            df_log[column] = np.log1p(df_log[column])

            print(f"Column '{column}' → log(1+x) transformation applied")

     return df_log
 
 
 
    #--------Six Method---------
    
    def sqrt_transform(self, columns=None):
        """
        Applies a square root transformation as a softer alternative to log.
        Less aggressive than log, good for moderately skewed distributions.
        Only applicable to non-negative columns.

        Args:
            columns: list of column names to transform.
                     If None, applies to all numeric columns.

        Returns:
            DataFrame with sqrt-transformed columns
        """
        df_sqrt = self.dataframe.copy()

        target_columns = columns if columns else self._get_numeric_columns()

        for column in target_columns:
            # Check that all values are non-negative (sqrt of negative = undefined in real numbers)
            if (df_sqrt[column] < 0).any():
                print(f"Column '{column}' skipped : contains negative values")
                continue

            df_sqrt[column] = np.sqrt(df_sqrt[column])

            print(f"Column '{column}' : sqrt transformation applied")

        return df_sqrt
            
            
    def treat(self, method, outlier_indices=None, **kwargs):
        """
        Main entry point for outlier treatment.
        Calls the appropriate method based on the chosen strategy.

        Args:
            method (str): one of 'remove', 'winsorize', 'impute_median',
                          'impute_mean', 'log', 'sqrt'
            outlier_indices: required for 'remove', 'impute_median', 'impute_mean'
            **kwargs: additional arguments passed to the chosen method
                      (e.g. lower_quantile, upper_quantile for winsorize)

        Returns:
            Treated DataFrame
        """
        if method == "remove":
            return self.remove_outliers(outlier_indices)
        elif method == "winsorize":
            return self.winsorize(**kwargs)
        elif method == "impute_median":
            return self.impute_median(outlier_indices)
        elif method == "impute_mean":
            return self.impute_mean(outlier_indices)
        elif method == "log":
            return self.log_transform(**kwargs)
        elif method == "sqrt":
            return self.sqrt_transform(**kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available methods: 'remove', 'winsorize', 'impute_median', "
                f"'impute_mean', 'log', 'sqrt'"
            )
            
    
    
        

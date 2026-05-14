import numpy as np
import pandas as pd 

class Multivariate_Treatment :
    """
    A multivariate outlier is a point that is anormal when considering
        MULTIPLE columns together, even if each individual value looks normal.

        Example: height=150cm + weight=120kg → each value alone seems ok,
                 but together they are anormal.
    
    
    This class handles the treatment of outliers in multivariate data.
    unlike univariate treatment , multivariate methods consider relationships between
    columns when deciding how to treat an outlier .
    
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe.copy()
        #We wor on a copy to never modify the original dataframe 
        
        
    def _get_numeric_columns(self):
        """Returns only the numeric columns of the dataframe."""
        return self.dataframe.select_dtypes(include=np.number).columns
    
    
    
    def remove_outliers(self, outlier_indices):
        """
        Removes entire rows identified as multivariate outliers.
        Args:
            outlier_indices: list or array of row indices to remove

        Returns:
            DataFrame with outlier rows dropped
        """
        cleaned_df = self.dataframe.drop(index=outlier_indices, errors="ignore")
        print(f"{len(outlier_indices)} multivariate outlier(s) removed. "
              f"Rows before: {len(self.dataframe)} → after: {len(cleaned_df)}")

        return cleaned_df
    
    
    def impute_median(self, outlier_indices):
       
        """
        Replaces outlier values with the median of each column,
        computed only on inlier rows.
        Each column is treated independently but only inlier rows
        are used to compute the reference median.

        Args:
            outlier_indices: list or array of row indices identified as outliers

        Returns:
            DataFrame with outlier rows replaced column by column by inlier medians
        """
        df_imputed = self.dataframe.copy()
         # Inlier mask: all rows that are NOT outliers
        inlier_mask = ~df_imputed.index.isin(outlier_indices)

        for column in self._get_numeric_columns():
            # Compute median using ONLY inlier rows for this column
            median_value = df_imputed.loc[inlier_mask, column].median()

            # Replace the value of outlier rows in this column with the median
            df_imputed.loc[outlier_indices, column] = median_value

            print(f"Column '{column}' → outliers replaced by inlier median = {median_value:.2f}")

        return df_imputed
        
        
    def winsorize(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        Caps extreme values column by column using quantile bounds.
        In the multivariate context, the same capping is applied consistently
        across all numeric columns, preserving the structure of the dataset.

        Args:
            lower_quantile (float): lower bound quantile. Defaults to 0.05
            upper_quantile (float): upper bound quantile. Defaults to 0.95

        Returns:
            DataFrame with all numeric columns capped
        """
        df_capped = self.dataframe.copy()

        for column in self._get_numeric_columns():
            lower_bound = df_capped[column].quantile(lower_quantile)
            upper_bound = df_capped[column].quantile(upper_quantile)

            # clip() brings values outside the bounds back to the bound value
            df_capped[column] = df_capped[column].clip(
                lower=lower_bound,
                upper=upper_bound
            )

            print(f"Column '{column}' → capped to "
                  f"[{lower_bound:.2f}, {upper_bound:.2f}]")

        return df_capped
    
    
    
    def impute_knn(self, outlier_indices, n_neighbors=5):
        """
        Replaces outlier values with the weighted average of the k nearest
        inlier neighbors, computed using ALL numeric columns simultaneously.

        This is the most powerful multivariate method because:
        - It uses the relationships between columns to find similar points
        - Closer neighbors have more influence than distant ones
        - The replaced value is coherent with the rest of the row

        Args:
            outlier_indices: list or array of row indices identified as outliers
            n_neighbors (int): number of neighbors to use. Defaults to 5

        Returns:
            DataFrame with outlier values replaced by weighted KNN average
        """
        df_imputed = self.dataframe.copy()
        
        for column in self._get_numeric_columns():
            df_imputed[column] = df_imputed[column].astype(float)

        # Separate inliers and outliers
        inlier_mask = ~df_imputed.index.isin(outlier_indices)
        df_inliers  = df_imputed.loc[inlier_mask, self._get_numeric_columns()]
        df_outliers = df_imputed.loc[outlier_indices, self._get_numeric_columns()]

        # Convert to numpy arrays for distance computation
        inliers_array  = df_inliers.values   
        outliers_array = df_outliers.values  

        # For each outlier row, find its k nearest inlier neighbors
        for i, outlier_idx in enumerate(outlier_indices):

            outlier_point = outliers_array[i]  # the outlier row as a 1D array

            # Step 1: compute Euclidean distance from this outlier to ALL inliers
            distances = self._euclidean_distances(outlier_point, inliers_array)

            # Step 2: find the indices of the k smallest distances
            knn_indices = np.argsort(distances)[:n_neighbors]

            # Step 3: get the distances of only the k nearest neighbors
            knn_distances = distances[knn_indices]

            # Step 4: compute weights  closer neighbors have more influence
            # We add a tiny epsilon to avoid division by zero if distance = 0
            epsilon = 1e-10
            weights = 1 / (knn_distances + epsilon)

            # Normalize weights so they sum to 1
            weights = weights / weights.sum()

            # Step 5: for each column, compute the weighted average of the k neighbors
            for column in self._get_numeric_columns():
                # Get the values of the k neighbors for this column
                neighbor_values = df_inliers.iloc[knn_indices][column].values

                # Weighted average: closer neighbors contribute more
                imputed_value = np.sum(weights * neighbor_values)

                # Replace the outlier value in this column
                df_imputed.loc[outlier_idx, column] = imputed_value

            print(f"Row {outlier_idx} → replaced using {n_neighbors} nearest neighbors")

        return df_imputed

    def _euclidean_distances(self, point, matrix):
        """
        Computes the Euclidean distance between one point and all rows of a matrix.
        This is the core distance function used by KNN imputation.
        This function use one of the most important notion  in Numpy: The vector broadcasting 
        Args:
            point: 1D array representing the outlier point (shape: n_features)
            matrix: 2D array of inlier points (shape: n_inliers x n_features)

        Returns:
            1D array of distances (shape: n_inliers)
        """
        # For each inlier row: sqrt( sum( (outlier_col - inlier_col)^2 ) )
        # matrix - point broadcasts automatically across all rows
        diff = matrix - point   # Automatically performs the subtraction between the outlier and the points in the inlier matrix.
        squared = diff ** 2            # square each difference
        summed = squared.sum(axis=1)   # Add up the values ​​found horizontally
        return np.sqrt(summed)         # square root → Euclidean distance
    
    
    
    
    def treat(self, method, outlier_indices=None, **kwargs):
        """
        Main entry point for multivariate outlier treatment.

        Args:
            method (str): one of 'remove', 'winsorize',
                          'impute_median', 'impute_knn'
            outlier_indices: required for 'remove', 'impute_median', 'impute_knn'
            **kwargs: additional arguments passed to the chosen method
                      e.g. n_neighbors=5 for impute_knn
                           lower_quantile, upper_quantile for winsorize

        Returns:
            Treated DataFrame
        """
        if method == "remove":
            return self.remove_outliers(outlier_indices)
        elif method == "winsorize":
            return self.winsorize(**kwargs)
        elif method == "impute_median":
            return self.impute_median(outlier_indices)
        elif method == "impute_knn":
            return self.impute_knn(outlier_indices, **kwargs)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available methods: 'remove', 'winsorize', "
                f"'impute_median', 'impute_knn'"
            )
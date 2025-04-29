#!/usr/bin/env python
# coding: utf-8

"""Handling missing values in datasets using different strategies."""

import numpy as np
import pandas as pd

class MissingValueHandler:
    """
    Class to handle missing values in datasets using various techniques
    (deletion, statistical imputation, default value, KNN, and regression).
    """

    def __init__(self):
        pass

    def _convert_to_dataframe(self, X):
        """
        Description:
            Converts input to a pandas DataFrame if not already one.

        Args:
            X (pd.DataFrame | np.ndarray | array-like): Input data.

        Returns:
            pd.DataFrame: Data converted to DataFrame.

        Example:
            >>> handler._convert_to_dataframe(np.array([[1, 2], [3, np.nan]]))
        """
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def remove_missing(self, X, threshold=0.5, axis=0):
        """
        Description:
            Removes rows or columns containing too many missing values.

        Args:
            X (pd.DataFrame | np.ndarray): Input data.
            threshold (float): Proportion threshold of NaN allowed before deletion (between 0 and 1).
            axis (int): Axis to process (0 for rows, 1 for columns).

        Returns:
            pd.DataFrame: Data after deletion.

        Example:
            >>> handler.remove_missing(df, threshold=0.4, axis=0)
        """
        df = self._convert_to_dataframe(X)
        df_cleaned = df.dropna(thresh=int(threshold * df.shape[axis]), axis=axis)
        return df_cleaned

    def impute_statistical(self, X, strategy="mean"):
        """
        Description:
            Imputes missing values using a statistical strategy.

        Args:
            X (pd.DataFrame | np.ndarray): Input data.
            strategy (str): Imputation method ('mean', 'median' or 'mode').

        Returns:
            pd.DataFrame: Imputed data.

        Example:
            >>> handler.impute_statistical(df, strategy="median")
        """
        df = self._convert_to_dataframe(X)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    def impute_default(self, X, value=0):
        """
        Description:
            Replaces all missing values with a constant value.

        Args:
            X (pd.DataFrame | np.ndarray): Input data.
            value (int | float): Value used to replace NaN.

        Returns:
            pd.DataFrame: Imputed data.

        Example:
            >>> handler.impute_default(df, value=-1)
        """
        df = self._convert_to_dataframe(X)
        return df.fillna(value)

    def impute_knn(self, X, k=3):
        """
        Description:
            Imputes missing values using k-nearest neighbors.

        Args:
            X (pd.DataFrame | np.ndarray): Input data.
            k (int): Number of neighbors to consider.

        Returns:
            pd.DataFrame: Imputed data.

        Example:
            >>> handler.impute_knn(df, k=5)
        """
        from scipy.spatial.distance import cdist

        df = self._convert_to_dataframe(X)

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                non_missing = df[df[col].notnull()]
                missing = df[df[col].isnull()]

                if len(non_missing) == 0:
                    continue

                distances = cdist(
                    missing.drop(col, axis=1).fillna(0),
                    non_missing.drop(col, axis=1).fillna(0),
                    metric='euclidean'
                )

                nearest_indices = np.argsort(distances, axis=1)[:, :k]
                knn_values = non_missing.iloc[nearest_indices.flatten()][col].values.reshape(-1, k)
                imputed_values = np.nanmean(knn_values, axis=1)

                df.loc[df[col].isnull(), col] = imputed_values
        return df

    def impute_regression(self, X, target_col):
        """
        Description:
            Imputes a target column by predicting it with a regression model
            based on other columns.

        Args:
            X (pd.DataFrame | np.ndarray): Input data.
            target_col (str): Name of column to impute.

        Returns:
            pd.DataFrame: Imputed data.

        Raises:
            ValueError: If target column doesn't exist in DataFrame.

        Example:
            >>> handler.impute_regression(df, target_col='Age')
        """
        from sklearn.linear_model import LinearRegression

        df = self._convert_to_dataframe(X)

        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' doesn't exist in X.")

        known = df[df[target_col].notnull()]
        unknown = df[df[target_col].isnull()]

        if len(unknown) == 0:
            return df

        model = LinearRegression()
        model.fit(known.drop(target_col, axis=1), known[target_col])
        predicted_values = model.predict(unknown.drop(target_col, axis=1))

        df.loc[df[target_col].isnull(), target_col] = predicted_values
        return df

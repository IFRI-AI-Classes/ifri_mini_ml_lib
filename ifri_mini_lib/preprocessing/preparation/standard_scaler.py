"""Standardisation"""
import numpy as np
import pandas as pd

class StandardScaler:
    """
    Description:
        A custom implementation of a standard scaler similar to scikit-learn's StandardScaler.

        This class centers and scales input data so that each feature has a mean of 0 and a standard deviation of 1.
        It supports both NumPy arrays and pandas DataFrames.

    Attributes:
        mean_ (np.ndarray): Mean of each feature in the training data.
        std_ (np.ndarray): Standard deviation of each feature in the training data.
        is_dataframe (bool): Whether the input was a pandas DataFrame.
        columns (Index or None): Column names of the DataFrame if applicable.
    """

    def __init__(self):
        """
        Description:
            Initializes the StandardScaler with default attributes.
        """
        self.mean_ = None
        self.std_ = None
        self.is_dataframe = False
        self.columns = None

    def _convert_to_array(self, X):
        """
        Description:
            Converts input data to a NumPy array if it's a DataFrame.

        Args:
            X (np.ndarray or pd.DataFrame): Input data.

        Returns:
            np.ndarray: Converted NumPy array.
        """
        if isinstance(X, pd.DataFrame):
            self.is_dataframe = True
            self.columns = X.columns
            return X.values
        self.is_dataframe = False
        return np.array(X)

    def _convert_to_dataframe(self, X_scaled):
        """
        Description:
            Converts the scaled NumPy array back to a pandas DataFrame if the original data was a DataFrame.

        Args:
            X_scaled (np.ndarray): Scaled data.

        Returns:
            pd.DataFrame or np.ndarray: Data in original input format.
        """
        if self.is_dataframe:
            return pd.DataFrame(X_scaled, columns=self.columns)
        return X_scaled

    def fit(self, X):
        """
        Description:
            Computes the mean and standard deviation of each feature from the input data.

        Args:
            X (np.ndarray or pd.DataFrame): Input data where each row is a sample.

        Example:
            scaler = StandardScaler()
            scaler.fit(data)
        """
        X = self._convert_to_array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)

    def transform(self, X):
        """
        Description:
            Applies standardization to the input data using the previously computed mean and std.

        Args:
            X (np.ndarray or pd.DataFrame): Input data to standardize.

        Returns:
            np.ndarray or pd.DataFrame: Standardized data.

        Raises:
            ValueError: If fit was not called before transform.

        Example:
            X_scaled = scaler.transform(data)
        """
        X = self._convert_to_array(X)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' before 'transform'.")

        X_scaled = (X - self.mean_) / self.std_
        return self._convert_to_dataframe(X_scaled)

    def fit_transform(self, X):
        """
        Description:
            Fits the scaler to the data and then transforms it.

        Args:
            X (np.ndarray or pd.DataFrame): Input data to fit and transform.

        Returns:
            np.ndarray or pd.DataFrame: Standardized data.

        Example:
            X_scaled = scaler.fit_transform(data)
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """
        Description:
            Reverses the standardization process and returns data to its original scale.

        Args:
            X_scaled (np.ndarray or pd.DataFrame): Standardized data.

        Returns:
            np.ndarray or pd.DataFrame: Data restored to original scale.

        Raises:
            ValueError: If fit was not called before inverse_transform.

        Example:
            original_data = scaler.inverse_transform(X_scaled)
        """
        X_scaled = self._convert_to_array(X_scaled)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' before 'inverse_transform'.")

        X_original = (X_scaled * self.std_) + self.mean_
        return self._convert_to_dataframe(X_original)

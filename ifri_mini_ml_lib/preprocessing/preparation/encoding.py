import pandas as pd
from typing import Union, List, Literal
import numpy as np


class CategoricalEncoder:
    """
    A flexible categorical encoder that supports multiple encoding techniques for categorical variables.

    This class provides functionality to encode categorical variables using various techniques:

    - Label Encoding: Assigns each unique category an integer value
    - Ordinal Encoding: Similar to label encoding but categories are sorted first
    - Frequency Encoding: Replaces categories with their frequency in the dataset
    - Target Encoding: Replaces categories with the mean of the target variable for that category
    - One-Hot Encoding: Creates binary columns for each category
        
    Args:
        encoding_type (str): Type of encoding to apply. Options: 'onehot', 'label', 'ordinal', 'frequency', 'target'. Default is 'onehot'.
            ('onehot' is deprecated . It will be removed soon. Use OneHotEncoder instead)
        target_column (str): Name of the target column (required for target encoding). Default is None.
    """
    
    def __init__(self, encoding_type='onehot', target_column=None):
        self.encoding_type = encoding_type
        self.target_column = target_column
        self.mapping = {}  # Stores encoding mappings

    def fit(self, X, y=None):
        """
        Learn the encoding mappings from the data. Computes and stores the necessary encoding information based on the training data.
            
        Args:
            X (pd.DataFrame): Input data containing categorical features to encode
            y (pd.Series, optional): Target variable (required for target encoding)
            
        Raises:
            ValueError: If target encoding is selected but no target variable is provided
        """
        if self.encoding_type == 'target' and y is None:
            raise ValueError("Target encoding requires target column `y`.")
        
        for column in X.select_dtypes(include=['object', 'string']).columns:
            if self.encoding_type == 'label':
                self.mapping[column] = {cat: idx for idx, cat in enumerate(X[column].unique())}
            elif self.encoding_type == 'ordinal':
                if column == 'size':
                    categories = ['S', 'M', 'L']
                    self.mapping[column] = {cat: idx for idx, cat in enumerate(categories)}
                else:
                    self.mapping[column] = {cat: idx for idx, cat in enumerate(sorted(X[column].unique()))}
            elif self.encoding_type == 'frequency':
                freq = X[column].value_counts(normalize=True)
                self.mapping[column] = freq.to_dict()
            elif self.encoding_type == 'target':
                target_means = X.join(y).groupby(column)[self.target_column].mean()
                self.mapping[column] = target_means.to_dict()
            elif self.encoding_type == 'onehot':
                # No mapping needed for one-hot encoding
                pass
            else:
                raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def transform(self, X):
        """
        Apply the encoding to new data using the learned mappings.
        Transforms the input data by applying the encoding learned during fit().
            
        Args:
            X (pd.DataFrame): Data to be encoded
            
        Returns:
            pd.DataFrame: Transformed data with categorical features encoded according to the specified method
        """
        X_encoded = X.copy()

        for column in X_encoded.select_dtypes(include=['object', 'string']).columns:
            if self.encoding_type in ['label', 'ordinal', 'frequency', 'target']:
                X_encoded[column] = X_encoded[column].map(self.mapping[column])
            elif self.encoding_type == 'onehot':
                dummies = pd.get_dummies(X_encoded[column], prefix=column)
                X_encoded = X_encoded.drop(column, axis=1)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)

        return X_encoded

    def fit_transform(self, X, y=None):
        """
        Learn the encoding and apply it to the training data in one step.
        Convenience method that combines fit() and transform() operations.
            
        Args:
            X (pd.DataFrame): Training data to fit and transform
            y (pd.Series, optional): Target variable (required for target encoding)
            
        Returns:
            pd.DataFrame: Transformed data with categorical features encoded
        """
        self.fit(X, y)
        return self.transform(X)

class OrdinalEncoder:
    """    
    Encodes categorical features as an integer array (0 to n_categories - 1). 
    Supports mixed data types and provides flexible strategies for handling unknown values 
    encountered during transformation.
    """

    def __init__(
        self, 
        categories: Union[Literal['auto'], List[List]] = 'auto', 
        handle_unknown: Literal['error', 'use_encoded_value'] = 'error', 
        unknown_value: int = -1
    ):
        """
        Initialize the encoder with specific behavior for categories and unknown values.

        :param categories: Source of categories for each feature.
            - 'auto' (str): Automatically determine categories from training data.
            - list of lists: Manually provided categories for each column.
        :param handle_unknown: Strategy for handling categories not seen during fitting.
            - 'error' (str): Raise a ValueError if an unknown category is found.
            - 'use_encoded_value' (str): Assign the value specified in `unknown_value`.
        :param unknown_value: The integer value to assign to unknown categories.
            - Type: int (defaults to -1).
        """
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categories_ = []

    def fit(self, X: np.ndarray) -> 'OrdinalEncoder':
        """
        Fit the OrdinalEncoder to the input data.

        Identifies and stores unique categories for each feature to build the 
        internal mapping.

        :param X: The data used to determine the categories.
            - Type: array-like of shape (n_samples, n_features).
        :return: The fitted encoder instance.
            - Type: OrdinalEncoder.
        """
        # Convert to object dtype to safely handle mixed types (int, str, etc.)
        X_temp = np.asarray(X, dtype=object)
        if X_temp.ndim != 2:
           raise ValueError("X must be 2D array")        
        
        n_features = X_temp.shape[1]
        self.categories_ = []

        for i in range(n_features):
            if self.categories == 'auto':
                # Extract unique values and sort them as strings for consistency
                col_cats = np.unique(X_temp[:, i])
            else:
                col_cats = np.array(self.categories[i])
            self.categories_.append(col_cats)
            
        if self.handle_unknown == "use_encoded_value":
           for cats in self.categories_:
               if 0 <= self.unknown_value < len(cats):
                  raise ValueError("unknown_value conflicts with valid category indices")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform categorical data into numerical codes.

        :param X: The data to transform.
            - Type: array-like of shape (n_samples, n_features).
        :return: Transformed numerical array.
            - Type: ndarray of type float64.
        :raises ValueError: If an unknown category is detected and handle_unknown is 'error'.
        """
        if not self.categories_:
           raise ValueError("This OrdinalEncoder instance is not fitted yet.")
        
        X_temp = np.asarray(X, dtype=object)
        X_out = np.empty(X_temp.shape, dtype=np.float64)
        
        if X_temp.shape[1] != len(self.categories_):
           raise ValueError("Number of features does not match fitted data")

        for i, cats in enumerate(self.categories_):
            # Create a lookup dictionary for O(1) average time complexity
            mapping = {val: idx for idx, val in enumerate(cats)}
            
            def encode_val(val):
                # Ensure type consistency during lookup
                val_native = val.item() if hasattr(val, 'item') else val
                val_lookup = val_native
                
                if val_lookup in mapping:
                    return mapping[val_lookup]
                
                if self.handle_unknown == 'error':
                    raise ValueError(f"Found unknown category '{val}' in column {i}")
                
                return self.unknown_value

            # Apply encoding logic across the entire column vector
            for j, val in enumerate(X_temp[:, i]):
              X_out[j, i] = encode_val(val)
            
        return X_out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it in a single step.

        :param X: The input data.
            - Type: array-like of shape (n_samples, n_features).
        :return: Transformed data.
            - Type: ndarray of type float64.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convert numerical codes back to their original categorical labels.

        :param X: The encoded numerical data.
            - Type: array-like of shape (n_samples, n_features).
        :return: Array of original categories.
            - Type: ndarray of type object.
        """
        if not self.categories_:
           raise ValueError("This OrdinalEncoder instance is not fitted yet.")
        X_temp = np.asarray(X, dtype=object)
        X_inv = np.empty(X_temp.shape, dtype=object)

        for i, cats in enumerate(self.categories_):
            col = X_temp[:, i]  
            col_float = col.astype(float)
            valid_mask = (col == self.unknown_value) | (np.floor(col_float) == col_float)              
            if not np.all(valid_mask):
                raise ValueError("Encoded values must be integers or unknown_value")

            indices = col.astype(int)

            mask = (indices >= 0) & (indices < len(cats)) & (indices != self.unknown_value)

            X_inv[mask, i] = cats[indices[mask]]
            X_inv[~mask, i] = None
            
        return X_inv

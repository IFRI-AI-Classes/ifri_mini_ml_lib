import pandas as pd

class CategoricalEncoder:
    """
    A flexible categorical encoder that supports multiple encoding techniques for categorical variables.
    
    Description:
        This class provides functionality to encode categorical variables using various techniques:
        - Label Encoding: Assigns each unique category an integer value
        - Ordinal Encoding: Similar to label encoding but categories are sorted first
        - Frequency Encoding: Replaces categories with their frequency in the dataset
        - Target Encoding: Replaces categories with the mean of the target variable for that category
        - One-Hot Encoding: Creates binary columns for each category
        
    Args:
        encoding_type (str): Type of encoding to apply. Options: 'onehot', 'label', 'ordinal', 'frequency', 'target'.
                            Default is 'onehot'.
        target_column (str): Name of the target column (required for target encoding). Default is None.
    """
    
    def __init__(self, encoding_type='onehot', target_column=None):
        self.encoding_type = encoding_type
        self.target_column = target_column
        self.mapping = {}  # Stores encoding mappings

    def fit(self, X, y=None):
        """
        Learn the encoding mappings from the data.
        
        Description:
            Computes and stores the necessary encoding information based on the training data.
            
        Args:
            X (pd.DataFrame): Input data containing categorical features to encode
            y (pd.Series, optional): Target variable (required for target encoding)
            
        Raises:
            ValueError: If target encoding is selected but no target variable is provided
        """
        if self.encoding_type == 'target' and y is None:
            raise ValueError("Target encoding requires target column `y`.")
        
        for column in X.select_dtypes(include='object').columns:
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
        
        Description:
            Transforms the input data by applying the encoding learned during fit().
            
        Args:
            X (pd.DataFrame): Data to be encoded
            
        Returns:
            pd.DataFrame: Transformed data with categorical features encoded according to the specified method
        """
        X_encoded = X.copy()

        for column in X_encoded.select_dtypes(include='object').columns:
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
        
        Description:
            Convenience method that combines fit() and transform() operations.
            
        Args:
            X (pd.DataFrame): Training data to fit and transform
            y (pd.Series, optional): Target variable (required for target encoding)
            
        Returns:
            pd.DataFrame: Transformed data with categorical features encoded
        """
        self.fit(X, y)
        return self.transform(X)
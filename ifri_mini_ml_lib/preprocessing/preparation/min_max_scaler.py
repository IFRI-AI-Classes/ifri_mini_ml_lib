import numpy as np
import pandas as pd

class MinMaxScaler:
    """
    Implementation of Min-Max scaling for data normalization.
    
    Scales features to a specified range by transforming each feature based on
    minimum and maximum values learned during fitting. The default target range
    is [0, 1], but can be customized during initialization.

    This scaler is sensitive to outliers since extreme values affect the min/max
    calculations. For robust scaling, consider using RobustScaler instead.
    """

    def __init__(self, feature_range=(0, 1)):
        """
        Initialize the MinMaxScaler with desired feature range.
        
        Args:
            feature_range (tuple): Desired range of transformed data (min, max). 
                                  Default: (0, 1)
            
        Returns:
            MinMaxScaler: Initialized scaler instance
            
        Example:
            >>> scaler = MinMaxScaler(feature_range=(0, 1))
            >>> scaler = MinMaxScaler(feature_range=(-1, 1))  # Custom range
        """
        self.min_ = None
        self.max_ = None
        self.range_min, self.range_max = feature_range
        self.columns = None 

    def fit(self, X):
        """
        Compute minimum and maximum values from training data for later scaling.
        
        Args:
            X (array-like): Training data to compute min/max values from. 
                           Can be list, numpy.ndarray, pandas.DataFrame, or pandas.Series
                           Shape should be (n_samples, n_features)
            
        Returns:
            self: Fitted scaler object
            
        Example:
            >>> scaler.fit(training_data)
            >>> scaler.fit([[1, 2], [3, 4]])  # With list input
        """
        X = self._convert_to_array(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        """
        Scale features using previously computed min/max values.
        
        Args:
            X (array-like): Data to be transformed (same shape as fit data)
            
        Returns:
            array-like: Transformed data in specified range [range_min, range_max]
                       Returns same type as input (DataFrame, Series or array)
            
        Raises:
            ValueError: If fit() hasn't been called first
            
        Example:
            >>> scaled_data = scaler.transform(new_data)
            >>> scaler.transform([[1, 2], [3, 4]])  # With list input
        """
        X_original = X  
        X = self._convert_to_array(X)

        if self.min_ is None or self.max_ is None:
            raise ValueError("Le scaler n'a pas été ajusté. Appelez 'fit' avant 'transform'.")

        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        return self._convert_back(X_scaled, X_original)

    def fit_transform(self, X):
        """
        Fit to data, then transform it in one step.
        
        Args:
            X (array-like): Data to fit and transform
            
        Returns:
            array-like: Transformed data in specified range
            
        Example:
            >>> scaled_data = scaler.fit_transform(data)
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """
        Transform scaled data back to original scale.
        
        Args:
            X_scaled (array-like): Scaled data to transform back to original scale
            
        Returns:
            array-like: Data in original scale (same type as input)
            
        Raises:
            ValueError: If fit() hasn't been called first
            
        Example:
            >>> original_data = scaler.inverse_transform(scaled_data)
        """
        X_original = X_scaled 
        X_scaled = self._convert_to_array(X_scaled)

        if self.min_ is None or self.max_ is None:
            raise ValueError("Le scaler n'a pas été ajusté. Appelez 'fit' avant 'inverse_transform'.")

        X_original_values = self.min_ + X_scaled * (self.max_ - self.min_)
        np.set_printoptions(suppress=True)
        return self._convert_back(X_original_values, X_original) 

    def _convert_to_array(self, X):
        """
        Convert input data to numpy array while preserving numerical values.
        
        Args:
            X (array-like): Input data to convert
            
        Returns:
            np.ndarray: Converted numpy array
            
        Note:
            Internal method not intended for direct use
        """
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns  # Stocke les colonnes
            return X.values
        if isinstance(X, pd.Series):
            self.columns = [X.name]  # Stocke le nom de la colonne (liste à 1 élément)
            return X.values
        self.columns = None
        return np.array(X)

    def _convert_back(self, X, original):
        """
        Convert numpy array back to original input format.
        
        Args:
            X (np.ndarray): Transformed numpy array
            original (array-like): Original input for format reference
            
        Returns:
            array-like: Data in original format
            
        Note:
            Internal method not intended for direct use
        """
        if isinstance(original, pd.DataFrame) and self.columns is not None:
            return pd.DataFrame(X, columns=self.columns)
        if isinstance(original, pd.Series) and self.columns:
            return pd.Series(X, name=self.columns[0])
        return X

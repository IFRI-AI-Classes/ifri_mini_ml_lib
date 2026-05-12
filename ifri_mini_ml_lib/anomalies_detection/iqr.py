"""
IQR (Interquartile Range) Anomaly Detector.

This module provides a robust statistical anomaly detection method based on the
Interquartile Range. It identifies data points that fall outside the
bounds defined by [Q1 - factor * IQR, Q3 + factor * IQR].
"""

import warnings
from typing import Union, List, Dict, Literal

import numpy as np


class IQR:
    """
    Anomaly detector based on the Interquartile Range (IQR) method.

    Description:
        The IQR method is a robust statistical technique for identifying outliers.
        For each feature, the algorithm computes the first quartile (Q1) and the
        third quartile (Q3). The Interquartile Range is defined as IQR = Q3 - Q1.
        A data point is flagged as an anomaly if any of its feature values falls
        below Q1 - factor * IQR or above Q3 + factor * IQR.

    Args:
        factor (float, optional): Multiplicative factor applied to the IQR to
            define the lower and upper bounds. Default is 1.5.
        handle_nan (str, optional): How to handle NaN values. 
            - 'raise': Raise a ValueError if NaNs are found.
            - 'omit': Use np.nanpercentile to ignore NaNs in calculation.
            Default is 'raise'.

    Attributes:
        Q1_ (np.ndarray): First quartile (25th percentile) per feature, set after fit().
        Q3_ (np.ndarray): Third quartile (75th percentile) per feature, set after fit().
        IQR_ (np.ndarray): Interquartile range per feature (Q3 - Q1), set after fit().
        lower_bound_ (np.ndarray): Lower fence per feature (Q1 - factor * IQR), set after fit().
        upper_bound_ (np.ndarray): Upper fence per feature (Q3 + factor * IQR), set after fit().

    Examples:
        >>> from ifri_mini_ml_lib.anomalies_detection import IQR
        >>> data = [[10], [12], [14], [15], [100]]
        >>> detector = IQR(factor=1.5)
        >>> detector.fit(data)
        >>> labels = detector.predict(data)
    """

    def __init__(self, factor: float = 1.5, handle_nan: Literal['raise', 'omit'] = 'raise') -> None:
        """
        Initializes the IQR anomaly detector.

        Args:
            factor (float): Multiplicative coefficient for the IQR. Default is 1.5.
            handle_nan (str): Strategy for NaN values ('raise' or 'omit'). Default is 'raise'.
        """
        if factor <= 0:
            raise ValueError(f"factor must be a positive number, got {factor}")
        if handle_nan not in ['raise', 'omit']:
            raise ValueError("handle_nan must be either 'raise' or 'omit'")

        self.factor = factor
        self.handle_nan = handle_nan

        self.Q1_ = None
        self.Q3_ = None
        self.IQR_ = None
        self.lower_bound_ = None
        self.upper_bound_ = None
        self._is_fitted = False

    def _validate_input(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """
        Internal helper for input validation and conversion.
        """
        X_arr = np.array(X, dtype=float)

        if X_arr.size == 0:
            raise ValueError("Input data must not be empty.")

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if X_arr.ndim != 2:
            raise ValueError(f"Input must be 1D or 2D array, got {X_arr.ndim}D.")

        if not np.issubdtype(X_arr.dtype, np.number):
             raise TypeError("Input data must be numeric.")

        if self.handle_nan == 'raise' and np.isnan(X_arr).any():
            raise ValueError("Input data contains NaN values. Set handle_nan='omit' to ignore them.")

        return X_arr

    def fit(self, X: Union[List, np.ndarray]) -> 'IQR':
        """
        Computes the IQR bounds from the training data.

        Args:
            X (array-like): Training data of shape (n_samples, n_features).

        Returns:
            self: The fitted IQR instance.
        """
        X_arr = self._validate_input(X)

        if X_arr.shape[0] < 2:
            raise ValueError("At least 2 samples are required to compute quartiles.")

        percentile_func = np.nanpercentile if self.handle_nan == 'omit' else np.percentile

        self.Q1_ = percentile_func(X_arr, 25, axis=0)
        self.Q3_ = percentile_func(X_arr, 75, axis=0)
        self.IQR_ = self.Q3_ - self.Q1_

        # Handle constant features
        constant_features = self.IQR_ == 0
        if np.any(constant_features):
            warnings.warn(
                f"Features {np.where(constant_features)[0]} are constant (IQR=0). "
                "These features will not contribute to anomaly detection.",
                UserWarning
            )

        self.lower_bound_ = self.Q1_ - self.factor * self.IQR_
        self.upper_bound_ = self.Q3_ + self.factor * self.IQR_

        self._is_fitted = True
        return self

    def predict(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """
        Predicts anomaly labels for the provided data.

        Args:
            X (array-like): Data to evaluate.

        Returns:
            np.ndarray: Labels (0 for normal, 1 for anomaly).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_arr = self._validate_input(X)

        if X_arr.shape[1] != self.lower_bound_.shape[0]:
            raise ValueError(f"Feature count mismatch: expected {self.lower_bound_.shape[0]}, got {X_arr.shape[1]}.")

        below = X_arr < self.lower_bound_
        above = X_arr > self.upper_bound_
        is_anomaly = np.any(below | above, axis=1)

        return is_anomaly.astype(int)

    def fit_predict(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """
        Fits the model and returns anomaly labels for X.

        Args:
            X (array-like): Data to fit and predict.

        Returns:
            np.ndarray: Anomaly labels.
        """
        return self.fit(X).predict(X)

    def get_bounds(self) -> Dict[str, np.ndarray]:
        """
        Returns the computed detection thresholds.

        Returns:
            dict: Dictionary with 'lower' and 'upper' bounds.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        return {"lower": self.lower_bound_, "upper": self.upper_bound_}

    def summary(self) -> None:
        """
        Prints a summary of the fitted parameters.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        
        print("="*40)
        print("         IQR Detector Summary")
        print("="*40)
        print(f"Factor: {self.factor}")
        print(f"Features: {len(self.Q1_)}")
        for i, (l, u) in enumerate(zip(self.lower_bound_, self.upper_bound_)):
            print(f"  Feature {i}: [{l:.4f}, {u:.4f}]")
        print("="*40)

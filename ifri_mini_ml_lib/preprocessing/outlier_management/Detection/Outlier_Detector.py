# outlier_detector.py

import numpy as np
from anomalies_detection.iqr import IQR
from anomalies_detection.isolation_forest import IsolationForest
from anomalies_detection.z_score import   ZScoreDetector
from .LOF import LOF


class OutlierDetector:
    """
    Unified interface for outlier detection.
    Wraps IQR, ZScore, IsolationForest (from the detection group)
    and LOF (implemented from scratch) under a single consistent API.

    All methods follow the same convention:
        0 = normal point
        1 = outlier (anomaly)

    Args:
        method (str): Detection method to use.
                      One of 'iqr', 'zscore', 'isolation_forest', 'lof'.
        **kwargs: Additional arguments passed to the underlying detector.
                  e.g. factor=1.5 for IQR, n_neighbors=20 for LOF.

    Examples:
        >>> detector = OutlierDetector(method="lof", n_neighbors=20)
        >>> detector.fit(X)
        >>> labels = detector.predict(X)
        >>> # or in one call:
        >>> labels = detector.fit_predict(X)
    """

    # Maps method name → class
    _DETECTORS = {
        "iqr":               IQR,
        "zscore":             ZScoreDetector,
        "isolation_forest":  IsolationForest,
        "lof":               LOF,
    }

    def __init__(self, method: str = "isolation_forest", **kwargs):
        if method not in self._DETECTORS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available methods: {list(self._DETECTORS.keys())}"
            )
        self.method = method

        # Instantiate the right detector with the given kwargs
        # e.g. OutlierDetector("iqr", factor=2.0) → IQR(factor=2.0)
        self._detector = self._DETECTORS[method](**kwargs)

        self._is_fitted = False

    def fit(self, X):
        """
        Fits the chosen detector on X.

        Args:
            X (array-like): Training data of shape (n_samples, n_features).

        Returns:
            self
        """
        self._detector.fit(X)
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predicts outlier labels for X.

        Args:
            X (array-like): Data to evaluate.

        Returns:
            np.ndarray: 1 = outlier, 0 = normal
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        result = self._detector.predict(X)
        
    
    
        if self.method == "zscore":
            result = np.array(result)
        if result.ndim == 2:
            # A row is an outlier if at least one feature is anomalous
            result = result.any(axis=1)
        return result.astype(int)

        return result

    def fit_predict(self, X) -> np.ndarray:
        """
        Fits the model and returns outlier labels for X in one call.

        Args:
            X (array-like): Data to fit and predict.

        Returns:
            np.ndarray: 1 = outlier, 0 = normal
        """
        return self.fit(X).predict(X)

    def score(self, X) -> np.ndarray:
        """
        Returns raw anomaly scores (not binary labels).
        Only available for IsolationForest and LOF.

        Args:
            X (array-like): Data to score.

        Returns:
            np.ndarray: Anomaly scores
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before score().")

        # IsolationForest uses anomaly_score(), LOF uses score_samples()
        if hasattr(self._detector, "anomaly_score"):
            return self._detector.anomaly_score(X)
        elif hasattr(self._detector, "score_samples"):
            return self._detector.score_samples(X)
        else:
            raise NotImplementedError(
                f"score() is not available for method '{self.method}'"
            )

    @property
    def detector(self):
        """
        Gives direct access to the underlying detector instance
        if the user needs specific attributes like bounds (IQR)
        or threshold (IsolationForest).
        """
        return self._detector
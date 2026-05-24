import numpy as np
from typing import Union, List, Tuple, Optional

# MODULE CONSTANTS
DEFAULT_THRESHOLD = 3.0
"""float : Default threshold for Z-score anomaly detection (commonly set to 3.0 for normal distributions)"""

MIN_DATA_POINTS = 5
"""int : Minimum number of data points required to perform Z-score detection (to ensure meaningful statistics)"""

MAD_TO_STD_FACTOR = 0.6745
"""float : Conversion factor from MAD to standard deviation for normal distribution"""


# modified_zscore_detection function

def modified_zscore_detection(
    data: Union[List[float], np.ndarray],
    threshold: float = 3.5,
    return_zscore: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    """
    Detects anomalies using the Modified Z-score method.

    Description:
        Computes anomaly scores based on the median and the
        Median Absolute Deviation (MAD). Unlike the classic
        Z-score, this method is robust to outliers.

    Args:
        data (list or np.ndarray): Input numerical data.
        threshold (float, optional): Detection threshold.
            Default is 3.5.
        return_zscore (bool, optional): If True, also returns
            the Modified Z-scores. Default is False.

    Returns:
        np.ndarray: Boolean array where True indicates anomalies.

        tuple (optional):
            anomalies (np.ndarray): Boolean anomaly mask.
            mz_scores (np.ndarray): Modified Z-scores.

    Examples:
        >>> data = [10, 12, 11, 10, 13, 100, 12, 11]
        >>> anomalies = modified_zscore_detection(data)
        
    """
    
    # Validation of parameters (same as in zscore_detection, but adapted to the context of median/MAD)
    if isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a list or numpy.ndarray, received {type(data)}")
    
    if len(data) < MIN_DATA_POINTS:
        raise ValueError(f"Insufficient data: need at least {MIN_DATA_POINTS} points")
    
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError(f"threshold must be a positive number, received {threshold}")
    
    # Robust statistics : median and MAD
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        raise ValueError(
                f"MAD is zero: all values are identical ({median}). "
                "Cannot detect anomalies with this method."

        )
    
    # Calcul of the Modified Z-scores and anomaly detection
    mz_scores = MAD_TO_STD_FACTOR * (data - median) / mad
    anomalies = np.abs(mz_scores) > threshold
    
    if return_zscore:
        return anomalies, mz_scores
    return anomalies

    
# class ZScoreDetector

class ZScoreDetector:
    
    """
    Z-score based anomaly detector.

    Description:
        Learns the mean and standard deviation from training
        data and detects anomalies using Z-scores.
        Points with |Z| greater than the threshold are
        classified as anomalies.

    Args:
        threshold (float, optional): Detection threshold.
            Default is 3.0.
        axis (int or None, optional): Axis used to compute
            statistics for 2D data. Default is None.

    Attributes:
        mean_ (float or np.ndarray): Learned mean values.
        std_ (float or np.ndarray): Learned standard deviations.
        is_fitted_ (bool): True if fit() has been called.

    Examples:
        >>> detector = ZScoreDetector(threshold=3.0)
        >>> detector.fit(X_train)
        >>> anomalies = detector.predict(X_test)
    """
    
    def __init__(self, 
                 threshold: float = DEFAULT_THRESHOLD,
                 axis: Optional[int] = None):
        self.threshold = threshold
        self.axis = axis
        self.mean_ = None
        self.std_ = None
        self.is_fitted_ = False
        
    # Training
    def fit(self,
            X_train: Union[List[float], np.ndarray]
            ) -> 'ZScoreDetector':
        
        """
        Learns the mean and standard deviation from training data.

        Description:
            Computes the statistics required for Z-score anomaly
            detection on future data.

        Args:
            X_train (list or np.ndarray): Training data of shape
                (n_samples,) or (n_samples, n_features).

        Returns:
            self: The fitted ZScoreDetector instance.

        Examples:
            >>> detector.fit([100, 102, 98, 101, 99])
        """
        
        if isinstance(X_train, list):
            X_train = np.array(X_train, dtype = float)
        elif isinstance(X_train, np.ndarray):
            X_train = X_train.astype(float)
        else:
            raise TypeError(f"X_train must be a list or numpy.ndarray, "
                            f"received {type(X_train).__name__}.")
            
        if X_train.size < MIN_DATA_POINTS:
            raise ValueError(f"Insufficient training data: at least {MIN_DATA_POINTS} "
                             f"points required, received {X_train.size}.")
            
        if X_train.ndim == 1:
            self.mean_ = np.mean(X_train)
            self.std_ = np.std(X_train, ddof = 1)
            
            if self.std_ == 0:
                raise ValueError(f"Standard deviation is zero: all training values are"
                                 f"identical ({self.mean_}). Cannot fit the detector.")
                
        else:
            self.mean_ = np.mean(X_train, axis = self.axis, keepdims = True)
            self.std_ = np.std(X_train, axis = self.axis, keepdims = True, ddof = 1)
            
            if np.any(self.std_ == 0):
                raise ValueError("Standard deviation is zero on at least one column or row "
                                 "of the training data. Check for constant features.")
        
        self.is_fitted_ = True
        return self
    
    # Prediction
    def predict(self,
                X_new: Union[List[float], np.ndarray],
                return_zscore: bool = False
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        """
        Detects anomalies in new data using Z-scores.

        Description:
            Computes Z-scores using the statistics learned during
            fit(). Values whose absolute Z-score exceeds the
            threshold are classified as anomalies.

        Args:
            X_new (list or np.ndarray): Input data to evaluate.
            return_zscore (bool, optional): If True, also returns
                the computed Z-scores. Default is False.

        Returns:
            np.ndarray: Boolean array where True indicates anomalies.

            tuple (optional):
                anomalies (np.ndarray): Boolean anomaly mask.
                z_scores (np.ndarray): Computed Z-scores.

        Examples:
            >>> detector.predict([101, 350, 99])
        """
        
        # Check fitted
        if not self.is_fitted_:
            raise RuntimeError("The detector has not been fitted yet. "
                                "Call fit(X_train) before predict().")
            
            
        # Validation
        if isinstance(X_new, list):
            X_new = np.array(X_new, dtype=float)
        elif isinstance(X_new, np.ndarray):
            X_new = X_new.astype(float)
        else:
            raise TypeError(f"X_new must be a list or numpy.ndarray, "
                            f"received {type(X_new).__name__}.")
            
        
        # Z-score computation
        z_scores = (X_new - self.mean_) / self.std_
        anomalies = np.abs(z_scores) > self.threshold
        
        if return_zscore:
            return anomalies, z_scores
        
        return anomalies
import numpy as np
from typing import Union, List, Tuple, Optional

# MODULE CONSTANTS
DEFAULT_THRESHOLD = 3.0
"""float: Default threshold for Z-score anomaly detection (commonly set to 3.0 for normal distributions)"""

MIN_DATA_POINTS = 5
"""int:Minimum number of data points required to perform Z-score detection (to ensure meaningful statistics)"""

MAD_TO_STD_FACTOR = 0.6745
"""float: Conversion factor from MAD to standard deviation for normal distribution"""


# modified_zscore_detection function

def modified_zscore_detection(
    data: Union[List[float], np.ndarray],
    threshold: float = 3.5,
    return_zscore: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Detects anomalies with the Modified Z-score (based on median and MAD).
    
    Unlike the classic Z-score, this version is ROBUST against
    the presence of outliers in the training data.

    
    Mathematical formula :
        M_i = 0.6745 × (x_i - median) / MAD
        
        where MAD = Median(|x_i - median|)
        and 0.6745 is a factor that makes the MAD comparable to the standard deviation
    
    Why is this version better ?
        - The median is not affected by extreme values
        - The MAD is more robust to outliers than the standard deviation
    
    Parameters
    ----------
    data : list or numpy.ndarray
        Array of numerical values to analyze.
        Must contain at least {MIN_DATA_POINTS} elements.
    
    threshold : float, default = 3.5
        Detection threshold. The value 3.5 is recommended by the literature
        (Iglewicz & Hoaglin, 1993).
    
    return_zscore : bool, default = False
        If True, also returns the Modified Z-scores.
    
    Returns
    -------
    anomalies : numpy.ndarray (bool)
        Array of booleans where True indicates an anomaly.

    mz_scores : numpy.ndarray (float), optional
        Returned only if return_zscore=True.
    
    Examples
    --------
    >>> data = [10, 12, 11, 10, 13, 100, 12, 11]
    >>> anomalies = modified_zscore_detection(data, threshold=3.5)
    >>> anomalies
    array([False, False, False, False, False, True, False, False])
    
    Notes
    -----
    This method is particularly recommended when :
        - The data already contains potential anomalies
        - The underlying distribution is not perfectly Gaussia
        - The distribution is not perfectly normal
        - The sample size is small (< 30 points)
    
    Reference
    ---------
    Iglewicz, B., & Hoaglin, D. C. (1993). 
    "How to Detect and Handle Outliers". ASQC Quality Press.
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
    Anomaly detector based on the Z-score, following a fit/predict pattern.
    
    Workflow
    --------
        1. fit(X_train)      : learn μ and σ from clean reference data.
        2. predict(X_new)    : flag anomalies in new data using learned stats.
        
    Parameters
    ----------
    threshold : float, default = 3.0
        Decision boundary above which |Z| flags a point as an anomaly.
        Recommanded values :
            - 2.5 : sensitive detection (higher recall, more false positives)
            - 3.0 : standard            (retains 99.7% of Gaussian data)
            - 4.0 : conservative        (lower recall, fewer false positives)
            
    axis : int or None, default = None
        Axis along which statistics are computed for 2D data.
            - None : global statistics across the entire matrix.
            - 0    : per-column statistics (recommanded).
            - 1    : per- row statistics.
        Ignored for 1D data.
        
    Attributes
    ----------
    mean_ : float or numpy.ndarray or None
        Sample mean learned during fit(). None before fit() is called.
    std_ : float or numpy.ndarray or None
        Sample standard deviation (ddof = 1) learned during fit().
        None before fit() is called.
    is_fitted_ : bool
    True once fit() has been successfully called.
        
    Examples
    --------
    Basic usage on 1D data :
    
    >>> X_train = [100, 102, 98, 101, 99, 103, 97, 101, 100, 102]
    >>> X_new   = [101, 99, 350, 98]
    >>> detector = ZScoreDetector(threshold=3.0)
    >>> detector.fit(X_train)
    >>> anomalies = detector.predict(X_new)
    >>> anomalies
    array([False, False,  True, False])

    2D usage with per-column statistics (axis=0) :

    >>> X_train = np.array([[37, 70], [36, 72], [37, 68],
    ...                     [36, 71], [37, 69]])
    >>> X_new   = np.array([[37, 71], [42, 180]])
    >>> detector = ZScoreDetector(threshold=3.0, axis=0)
    >>> detector.fit(X_train)
    >>> detector.predict(X_new)
    array([[False, False],
           [ True,  True]])
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
        Learn the mean and standard deviation from clean training data.
        Parameters
        ----------
        X_train : list or numpy.ndarray
            Reference data considered free of anomalies.
            - 1D : flat array of scalar values.
            - 2D : matrix of shape (n_samples, n_features).
            Must contain at least MIN_DATA_POINTS elements.
            
        Returns
        -------
        self : ZScoreDetector
            The fitted detector (allows method chaining).
            
        Raises
        ------
        TypeError
            If X_train is not a list or numpy.ndarray.
        ValueError
            - If X_train contains fewer than MIN_DATA_POINTS elements.
            - If the standard deviation is zero (constant feature or data);
        
        Examples
        --------
        >>> detector = ZScoreDetector(threshold = 3.0)
        >>> detector.fit([100, 102, 98, 101, 99, 103, 97, 101, 100, 102])
        >>> detector.is_fitted_
        True
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
        Detect anomalies in new data using the statistics learned during fit(). 
        
        Parameters
        ----------
        X_new : list or numpy.ndarray
            New data to evaluate.
            Must have the same number of features (columns) as X_train.
        return_zscore : bool, default = False
            If True, also returns the computed Z-scores.

        Returns
        -------
        anomalies : numpy.ndarray of bool
            Boolean mask where True indicates an anomaly.
        z_scores : numpy.ndarray of float, optional
            Returned only if return_zscore = True.

        Raises
        ------
        RuntimeError
            If predict() is called before fit().
        TypeError
            If X_new is not a list or numpy.ndarray.

        Examples
        --------
        >>> detector = ZScoreDetector( threshold = 3.0)
        >>> detector.fit([100, 102, 98, 101, 99, 103, 97, 101, 100, 102])
        >>> detector.predict([101, 350, 99])
        array([False,  True, False])
        
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
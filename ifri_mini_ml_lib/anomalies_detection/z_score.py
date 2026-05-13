import numpy as np
from typing import Union, List, Tuple, Optional

# MODULE CONSTANTS
DEFAULT_THRESHOLD = 3.0
"""float: Default threshold for Z-score anomaly detection (commonly set to 3.0 for normal distributions)"""

MIN_DATA_POINTS = 5
"""int:Minimum number of data points required to perform Z-score detection (to ensure meaningful statistics)"""

MAD_TO_STD_FACTOR = 0.6745
"""float: Conversion factor from MAD to standard deviation for normal distribution"""


# PRINCIPAL FUNCTION : CLASIC Z-SCORE DETECTION
def zscore_detection(
    data: Union[List[float], np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
    axis: Optional[int] = None,
    return_zscore: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:


    """
    Detects anomalies in a dataset using the Z-score.
    
    The Z-score measures how many standard deviations a value is from the 

    sample mean. A value is considered an anomaly when |Z| > threshold.
    Supports both 1D arrays and 2D matrices; for 2D data, statistics can be 
    computed globally or along the specified axis.
    
    Mathematical formula :
    For 1D data :
        Z_i = (x_i - μ) / σ  
              
    For 2D data with axis = 0 (per column / per feature) :
        Z_ij = (x_ij - μ_j) / σ_j
        
    For 2D data with axis = None (global, across entire matrix) :
        Z_ij = (x_ij - μ_global) / σ_global
        
    with μ = mean, σ = standard deviation.
 
    Parameters
    ----------
    data : list or numpy.ndarray
        Array of numerical values to analyze.

        - 1D : a flat list or 1-D array of scalar values.
        - 2D : a matrix of shape (n_samples, n_features).
        Must contain at least {MIN_DATA_POINTS} elements.
        
    threshold : float, default = 3.0
        Threshold above which a value is considered an anomaly.
        Recommended values :
            - 2.5 : sensitive detection (many anomalies : higher recall,
                    more false positives)
            - 3.0 : standard (recommended, 99.7% of Gaussian data)
            - 4.0 : very strict detection (few anomalies : lower recall,
                    fewer false positives)    
                             
    axis : int or None, default = None
        Axis along which mean and std are computed. Applies to 2D data only;
        ignored for 1D arrays.
            - None : global statistics across the entire matrix.
            - 0    : per-column statistics (recommended for ML feature matrices,
                     where each column is an independent feature).
            - 1    : per-row statistics.

        Must contain at least {MIN_DATA_POINTS} elements.
    
    Returns
    -------
    anomalies : numpy.ndarray (bool)
        Array of booleans where True indicates an anomaly.
    
    z_scores : numpy.ndarray (float), optional
        Returned only if return_zscore=True.
        The Z_scores corresponding to each data point.
    Raises
    -----
    ValueError
        - If data is empty or contains too few elements
        - If threshold is not a strictly positive number
        - If the standard deviation is zero (all values are identical -> Z-score undefined)
        - If *axis* is not 0, 1 or None when *data* is 2D.


    Examples
    --------
    >>> data = [10, 12, 11, 10, 13, 100, 12, 11]
    >>> anomalies = zscore_detection(data)
    >>> anomalies
    array([False, False, False, False, False, True, False, False])
           
    Notes
    -----
    **Sensitivity to existing outliers** : the classic Z-score uses the
    sample mean and standard deviation, both of which are themselves
    distorted by extreme values. As a result, a heavily contaminated
    dataset may mask some anomalies (masking effect) or inflate scores of
    normal points (swamping effect). For contaminated data, prefer
    ``modified_zscore_detection()``.
 
    **Choosing axis for 2D data** :
        - ``axis=0`` is the standard choice in machine learning when columns
          represent independent features measured on different scales.
          Each feature is standardized independently.
        - ``axis=None`` is appropriate only when all features share the same
          physical unit and scale.
 
    **Sample vs. population std** : ``ddof=1`` is used so the estimator is
    unbiased for the true population standard deviation when working with a
    finite sample.
    
    This classic Z_score is sensitive to anomalies themselves.
    For robust usage against already polluted data, prefer
    the function `modified_zscore_detection()`.
    """
    
    # 1. PARAMETER VALIDATION (Robust Validation)
    # Conversion and type validation
    if isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a list or numpy.ndarray, received {type(data)}")
    
    # Size verification
    if len(data) < MIN_DATA_POINTS:
        raise ValueError(
            f"Insufficient data: at least {MIN_DATA_POINTS} points required, "
            f"received {len(data)}"
        )
    
    # Validation of threshold
    if not isinstance(threshold, (int, float)) or threshold <= 0:
        raise ValueError(f"threshold must be a positive number, received {threshold}")
    
    #Calcul of mean and standard deviation
    mean = np.mean(data)      # μ (mean)
    std = np.std(data)        # σ (standard deviation)
    
    # Particular cases  : all values are identical → std = 0 → impossible to detect anomalies
    if std == 0:
        raise ValueError(
            f"Standard deviation is zero: all values are identical ({mean}). "
            "Impossible to detect anomalies with Z-score."
        )
    
    # Calcul of Z-scores and anomaly detection
    z_scores = (data - mean) / std
    
    # Absolute value then comparison with the threshold
    anomalies = np.abs(z_scores) > threshold
    
    if return_zscore:
        return anomalies, z_scores
    return anomalies



"""Now ,it's important to note that the classic Z-score is sensitive to the presence of anomalies in the data,
because the mean and standard deviation can be heavily influenced by extreme values.
For this reason, a more robust version of the Z-score, called the Modified Z-score,
is often recommended when the data may already contain anomalies or when the distribution is not perfectly normal.
The Modified Z-score is based on the median and the Median Absolute Deviation (MAD),
which are more robust statistics that are less affected by outliers.
The mathematical formula for the Modified Z-score is :
    M_i = 0.6745 × (x_i - median) / MAD
    
    where MAD = Median(|x_i - median|)
    and 0.6745 is a factor that makes the MAD comparable to the standard deviation for normal distributions."""

# SECONDARY FUNCTION : MODIFIED Z-SCORE (ROBUST)

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


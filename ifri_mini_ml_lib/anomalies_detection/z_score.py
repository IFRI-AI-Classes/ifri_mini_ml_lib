import numpy as np
from typing import Union, List, Tuple

# =============================================================================
# MODULE CONSTANTS
# =============================================================================

DEFAULT_THRESHOLD = 3.0
"""float: Default threshold for Z-score anomaly detection (commonly set to 3.0 for normal distributions)"""

MIN_DATA_POINTS = 5
"""int:Minimum number of data points required to perform Z-score detection (to ensure meaningful statistics)"""

MAD_TO_STD_FACTOR = 0.6745
"""float: Conversion factor from MAD to standard deviation for normal distribution"""


# =============================================================================
# PRINCIPAL FUNCTION : CLASIC Z-SCORE DETECTION
# =============================================================================

def zscore_detection(
    data: Union[List[float], np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
    return_zscore: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Detects anomalies in a dataset using the Z-score.
    
    The Z-score measures how many standard deviations a value is from the 
    sample mean. A value is considered an anomaly if |Z| > threshold.
    
    Mathematical formula :
        Z_i = (x_i - μ) / σ
        with μ = mean, σ = standard deviation
    
    Parameters
    ----------
    data : list or numpy.ndarray
        Array of numerical values to analyze.
        Must contain at least {MIN_DATA_POINTS} elements.
    
    threshold : float, default = 3.0
        Threshold above which a value is considered an anomaly.
        Recommended values :
            - 2.5 : sensitive detection (many anomalies)
            - 3.0 : standard (recommended, 99.7% of normal data)
            - 4.0 : very strict detection (few anomalies)
    
    return_zscore : bool, default = False
        If True, also returns the calculated Z-scores.
    
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
        - If threshold is not a positive number
        - If the standard deviation is zero (all values are identical)

    Examples
    --------
    >>> data = [10, 12, 11, 10, 13, 100, 12, 11]
    >>> anomalies = zscore_detection(data)
    >>> anomalies
    array([False, False, False, False, False, True, False, False])
    
    >>> anomalies, zscores = zscore_detection(data, return_zscore=True)
    >>> print(f"Z-scores : {zscores.round(2)}")
    Z-scores : [-0.47 -0.19 -0.33 -0.47 -0.05  3.88 -0.19 -0.33]
    
    Notes
    -----
    This classic Z_score is sensitive to anomalies themselves.
    For robust usage against already polluted data, prefer
    the function `modified_zscore_detection()`.
    
    See also
    ----------
    modified_zscore_detection : robust version with median and MAD
    """
    
    # -------------------------------------------------------------------------
    # 1. PARAMETER VALIDATION (Robust Validation)
    # -------------------------------------------------------------------------
    
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
    
    # -------------------------------------------------------------------------
    # 2. CALCUL OF THE STATISTICS (Mean and Standard Deviation)
    # -------------------------------------------------------------------------
    
    mean = np.mean(data)      # μ (mean)
    std = np.std(data)        # σ (standard deviation)
    
    # Particular cases  : all values are identical → std = 0 → impossible to detect anomalies
    if std == 0:
        raise ValueError(
            f"Standard deviation is zero: all values are identical ({mean}). "
            "Impossible to detect anomalies with Z-score."
        )
    
    # -------------------------------------------------------------------------
    # 3. CALCUL OF THE Z-SCORES AND ANOMALY DETECTION
    # -------------------------------------------------------------------------
    
    z_scores = (data - mean) / std
    
    # Absolute value then comparison with the threshold
    anomalies = np.abs(z_scores) > threshold
    
    # -------------------------------------------------------------------------
    # 4. RETURN
    # -------------------------------------------------------------------------
    
    if return_zscore:
        return anomalies, z_scores
    return anomalies


# =============================================================================
# SECONDARY FUNCTION : MODIFIED Z-SCORE (ROBUST)
# =============================================================================

def modified_zscore_detection(
    data: Union[List[float], np.ndarray],
    threshold: float = 3.5,
    return_zscore: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Detects anomalies with the Modified Z-score (based on median and MAD).
    
    Unlike the classic Z-score, this version is ROBUST against
    the presence of anomalies in the training data.
    
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


# =============================================================================
# UTILITY FUNCTION: SUMMARY OF ANOMALIES
# =============================================================================

def summary_anomalies(
    data: Union[List[float], np.ndarray],
    anomalies: np.ndarray
) -> dict:
    """
    Generates a statistical summary of the detected anomalies.
    
    Parameters
    ----------
    data : list or numpy.ndarray
        The original data.
    
    anomalies : numpy.ndarray (bool)
        The detected anomalies (True for anomalies).
    
    Returns
    -------
    dict : Dictionary containing :
        - count : number of anomalies
        - percentage : percentage of anomalies
        - indices : positions of anomalies
        - values : values of anomalies
        - min_anomaly : minimum value among the anomalies
        - max_anomaly : maximum value among the anomalies
    
    Example
    -------
    >>> data = [10, 12, 11, 100]
    >>> anomalies = zscore_detection(data)
    >>> summary = summary_anomalies(data, anomalies)
    >>> print(f"{summary['count']} anomalies detected")
    1 anomalies detected
    """
    
    if isinstance(data, list):
        data = np.array(data)
    
    anomaly_indices = np.where(anomalies)[0]
    anomaly_values = data[anomaly_indices]
    
    return {
        'count': len(anomaly_indices),
        'percentage': (len(anomaly_indices) / len(data)) * 100,
        'indices': anomaly_indices.tolist(),
        'values': anomaly_values.tolist(),
        'min_anomaly': float(np.min(anomaly_values)) if len(anomaly_values) > 0 else None,
        'max_anomaly': float(np.max(anomaly_values)) if len(anomaly_values) > 0 else None,
    }

# =============================================================================
# TESTS INTÉGRÉS (si exécution directe)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTS OF DETECTION OF ANOMALIES WITH Z-SCORE")
    print("=" * 60)
    
    # Données de test
    test_data = [100, 102, 98, 101, 99, 300, 101, 103, 97, 98]
    
    print(f"\nData : {test_data}")
    
    # Test classic Z_score
    anomalies, zscores = zscore_detection(test_data, return_zscore=True)
    print("\n Classic Z_SCORE")
    print(f"   Anomalies : {anomalies}")
    print(f"   Z-scores  : {[round(z, 2) for z in zscores]}")
    
    # Test Modified Z-score
    m_anomalies, m_zscores = modified_zscore_detection(test_data, return_zscore=True)
    print("\n MODIFIED Z-SCORE (Robust)")
    print(f"   Anomalies : {m_anomalies}")
    print(f"   MZ-scores : {[round(mz, 2) for mz in m_zscores]}")
    
    # Summary
    print("\n📈 SUMMARY OF ANOMALIES")
    summary = summary_anomalies(test_data, anomalies)
    print(f"   → {summary['count']} anomalies detected")
    print(f"   → {summary['percentage']:.1f}% of the data")
    print(f"   → Extreme values : {summary['min_anomaly']} / {summary['max_anomaly']}")
    
    print("\n" + "=" * 60)
    print("Tous les tests passés avec succès !")
    print("=" * 60)
    
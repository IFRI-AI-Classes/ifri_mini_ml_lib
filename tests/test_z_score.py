import pytest
import numpy as np
from ifri_mini_ml_lib.anomalies_detection.z_score import (
    modified_zscore_detection,
    ZScoreDetector,
    DEFAULT_THRESHOLD,
    MIN_DATA_POINTS,
)
# FIXTURES — shared data across all tests

@pytest.fixture
def clean_1d():
    """1D array of normally distributed values — no anomalies."""
    return [100, 102, 98, 101, 99, 103, 97, 101, 100, 102]


@pytest.fixture
def anomalous_1d():
    """1D array with one clear anomaly (300 at index 5)."""
    return [100, 102, 98, 101, 99, 300, 101, 103, 97, 98]


@pytest.fixture
def clean_2d():
    """2D matrix (5 samples x 3 features) — no anomalies."""
    return np.array([
        [37.2, 72.0, 0.9],
        [36.8, 68.0, 1.1],
        [37.0, 70.0, 1.0],
        [36.9, 71.0, 0.95],
        [37.1, 69.0, 1.05],
    ])


@pytest.fixture
def anomalous_2d():
    """2D matrix (5 samples x 3 features) with one anomalous row (index 2)."""
    return np.array([
        [37.2,  72.0, 0.9],
        [36.8,  68.0, 1.1],
        [42.0, 180.0, 8.5],   # <- clear anomaly on all features
        [37.0,  70.0, 1.0],
        [36.9,  71.0, 0.95],
    ])


# 1. TESTS — modified_zscore_detection()

class TestModifiedZscoreDetection:
    """
    Tests for the modified_zscore_detection() function.

    This function detects anomalies using the robust Modified Z-score,
    based on the median and MAD instead of mean and std.
    It is more resistant to contaminated data than the classic Z-score.

    Covers : basic detection, threshold sensitivity, return of MZ-scores,
             input types, output shape, and parameter validation.
    """

    # 1.1  Basic detection

    def test_detects_anomaly_in_1d(self, anomalous_1d):
        """
        Purpose  : verify that the clear outlier (300 at index 5) is flagged.
        Expected : result[5] == True.
        Obtained : modified Z-score of 300 >> threshold = 3.5 → flagged.
        """
        result = modified_zscore_detection(anomalous_1d, threshold = 3.5)
        assert result[5] is np.bool_(True), \
            "Value 300 at index 5 should be flagged as an anomaly."

    def test_no_anomaly_on_clean_data(self, clean_1d):
        """
        Purpose  : verify that a clean dataset produces no anomalies.
        Expected : all False.
        Obtained : all MZ-scores well below threshold=3.5.
        """
        result = modified_zscore_detection(clean_1d, threshold = 3.5)
        assert not np.any(result), \
            "No anomalies expected in clean data."

    def test_output_shape_matches_input(self, anomalous_1d):
        """
        Purpose  : verify output shape matches input length.
        Expected : shape == (10,).
        Obtained : boolean array of same length as input.
        """
        result = modified_zscore_detection(anomalous_1d)
        assert result.shape == (len(anomalous_1d),)

    def test_output_dtype_is_bool(self, clean_1d):
        """
        Purpose  : verify that the returned array contains booleans.
        Expected : result.dtype == bool.
        Obtained : numpy boolean array.
        """
        result = modified_zscore_detection(clean_1d)
        assert result.dtype == bool

    # 1.2  Threshold sensitivity
    
    def test_lower_threshold_detects_more(self, anomalous_1d):
        """
        Purpose  : verify that lowering the threshold increases detections.
        Expected : sum(result_low) >= sum(result_high).
        Obtained : more anomalies flagged with threshold=2.0 vs threshold = 3.5.
        """
        result_high = modified_zscore_detection(anomalous_1d, threshold = 3.5)
        result_low  = modified_zscore_detection(anomalous_1d, threshold = 2.0)
        assert np.sum(result_low) >= np.sum(result_high)

    # 1.3  return_zscore

    def test_returns_mzscores_when_requested(self, anomalous_1d):
        """
        Purpose  : verify that return_zscore=True returns a tuple
                   (anomalies, mz_scores) with matching shapes.
        Expected : both arrays have shape (10,).
        Obtained : tuple of two numpy arrays of the same length.
        """
        anomalies, mz_scores = modified_zscore_detection(
            anomalous_1d, return_zscore = True
        )
        assert anomalies.shape == mz_scores.shape
        assert len(mz_scores) == len(anomalous_1d)

    def test_mzscore_of_outlier_exceeds_threshold(self, anomalous_1d):
        """
        Purpose  : verify that the MZ-score of the outlier (300) exceeds
                   the threshold, confirming why it is flagged.
        Expected : |mz_scores[5]| > 3.5.
        Obtained : MZ-score of 300 ≈ 67.28 >> 3.5.
        """
        _, mz_scores = modified_zscore_detection(
            anomalous_1d, return_zscore = True
        )
        assert abs(mz_scores[5]) > 3.5

    # 1.4  Input types

    def test_accepts_python_list(self):
        """
        Purpose  : verify that a plain Python list is accepted.
        Expected : no TypeError, result is a numpy array.
        Obtained : successful execution.
        """
        data   = [10, 12, 11, 13, 10, 12, 11, 10, 13, 12]
        result = modified_zscore_detection(data)
        assert isinstance(result, np.ndarray)

    def test_accepts_numpy_array(self):
        """
        Purpose  : verify that a numpy array is accepted.
        Expected : no TypeError, result is a numpy array.
        Obtained : successful execution.
        """
        data   = np.array([10, 12, 11, 13, 10, 12, 11, 10, 13, 12], dtype=float)
        result = modified_zscore_detection(data)
        assert isinstance(result, np.ndarray)

    # 1.5  Parameter validation

    def test_invalid_type_raises_typeerror(self):
        """
        Purpose  : verify that an invalid input type raises TypeError.
        Expected : TypeError.
        Obtained : TypeError because input is a string.
        """
        with pytest.raises(TypeError):
            modified_zscore_detection("not valid")

    def test_too_few_points_raises_valueerror(self):
        """
        Purpose  : verify that fewer than MIN_DATA_POINTS raises ValueError.
        Expected : ValueError.
        Obtained : ValueError because only 3 points provided.
        """
        with pytest.raises(ValueError):
            modified_zscore_detection([1, 2, 3])

    def test_invalid_threshold_raises_valueerror(self, clean_1d):
        """
        Purpose  : verify that a non-positive threshold raises ValueError.
        Expected : ValueError for threshold=0 and threshold=-1.
        Obtained : ValueError in both cases.
        """
        with pytest.raises(ValueError):
            modified_zscore_detection(clean_1d, threshold = 0)

        with pytest.raises(ValueError):
            modified_zscore_detection(clean_1d, threshold = -1.0)

    def test_zero_mad_raises_valueerror(self):
        """
        Purpose  : verify that constant data (MAD = 0) raises ValueError
                   because the Modified Z-score is mathematically undefined.
        Expected : ValueError.
        Obtained : ValueError because MAD=0 → division by zero.
        """
        with pytest.raises(ValueError):
            modified_zscore_detection([5, 5, 5, 5, 5, 5, 5])


# 2. TESTS — ZScoreDetector (fit / predict pattern)

class TestZScoreDetector:
    """
    Tests for the ZScoreDetector class (sklearn-style fit/predict API).

    This class separates the learning phase (fit) from the detection phase
    (predict), making it suitable for production systems and real-time
    monitoring where statistics are learned once on clean reference data.

    Covers : initialization, fit (1D and 2D with axis = 0/1/None),
             predict (1D and 2D), fit_predict, method chaining,
             and full error handling.
    """

    # 2.1  Initialization

    def test_default_initialization(self):
        """
        Purpose  : verify that all default attributes are set correctly.
        Expected : threshold = 3.0, axis = None, mean_ = None,
                   std_ = None, is_fitted_ = False.
        Obtained : all attributes match defaults.
        """
        detector = ZScoreDetector()
        assert detector.threshold  == DEFAULT_THRESHOLD
        assert detector.axis       is None
        assert detector.mean_      is None
        assert detector.std_       is None
        assert detector.is_fitted_ is False

    def test_custom_initialization(self):
        """
        Purpose  : verify that custom constructor parameters are stored.
        Expected : threshold = 2.5, axis = 0.
        Obtained : attributes reflect the passed values.
        """
        detector = ZScoreDetector(threshold = 2.5, axis = 0)
        assert detector.threshold == 2.5
        assert detector.axis      == 0

    # 2.2  fit() — 1D

    def test_fit_1d_sets_is_fitted(self, clean_1d):
        """
        Purpose  : verify that fit() sets is_fitted_ to True.
        Expected : is_fitted_ == True after fit().
        Obtained : flag correctly set.
        """
        detector = ZScoreDetector()
        detector.fit(clean_1d)
        assert detector.is_fitted_ is True

    def test_fit_1d_stores_correct_mean(self, clean_1d):
        """
        Purpose  : verify that fit() stores the correct mean.
        Expected : detector.mean_ ≈ np.mean(clean_1d).
        Obtained : values match within floating-point tolerance.
        """
        detector = ZScoreDetector()
        detector.fit(clean_1d)
        assert np.isclose(detector.mean_, np.mean(clean_1d))

    def test_fit_1d_stores_correct_std(self, clean_1d):
        """
        Purpose  : verify that fit() stores the correct std (ddof = 1).
        Expected : detector.std_ ≈ np.std(clean_1d, ddof = 1).
        Obtained : values match within floating-point tolerance.
        """
        detector = ZScoreDetector()
        detector.fit(clean_1d)
        assert np.isclose(detector.std_, np.std(clean_1d, ddof = 1))

    def test_fit_returns_self_for_chaining(self, clean_1d):
        """
        Purpose  : verify that fit() returns self to allow method chaining.
        Expected : return value is the same ZScoreDetector instance.
        Obtained : identity check passes.
        """
        detector = ZScoreDetector()
        result   = detector.fit(clean_1d)
        assert result is detector

    # 2.3  fit() — 2D

    def test_fit_2d_axis0_shape(self, clean_2d):
        """
        Purpose  : verify that fit() stores mean_ and std_ with correct shapes
                   for 2D data with axis = 0 (per-column statistics).
        Expected : mean_.shape == (1, 3), std_.shape == (1, 3).
        Obtained : keepdims=True produces shape (1, n_features).
        """
        detector = ZScoreDetector(axis = 0)
        detector.fit(clean_2d)
        assert detector.mean_.shape == (1, 3)
        assert detector.std_.shape  == (1, 3)

    def test_fit_2d_axis1_shape(self, clean_2d):
        """
        Purpose  : verify that fit() stores mean_ and std_ with correct shapes
                   for 2D data with axis = 1 (per-row statistics).
        Expected : mean_.shape == (5, 1), std_.shape == (5, 1).
        Obtained : keepdims = True produces shape (n_samples, 1).
        """
        detector = ZScoreDetector(axis = 1)
        detector.fit(clean_2d)
        assert detector.mean_.shape == (5, 1)
        assert detector.std_.shape  == (5, 1)

    def test_fit_2d_axis_none_sets_is_fitted(self, clean_2d):
        """
        Purpose  : verify that fit() works with axis = None on 2D data
                   (global statistics across the entire matrix).
        Expected : is_fitted_ == True.
        Obtained : global mean and std computed successfully.
        """
        detector = ZScoreDetector(axis = None)
        detector.fit(clean_2d)
        assert detector.is_fitted_ is True

    # 2.4  predict() — 1D

    def test_predict_1d_no_anomaly(self, clean_1d):
        """
        Purpose  : verify that predict() returns all False when new data
                   is similar to the training data.
        Expected : all False.
        Obtained : all Z-scores well below threshold=3.0.
        """
        detector = ZScoreDetector(threshold = 3.0)
        detector.fit(clean_1d)
        result = detector.predict(clean_1d)
        assert not np.any(result)

    def test_predict_1d_detects_outlier(self, clean_1d):
        """
        Purpose  : verify that a clear outlier (350) in new data is detected
                   using statistics learned from clean training data.
        Expected : result[2] == True (index of 350 in X_new).
        Obtained : Z-score of 350 >> threshold = 3.0 → flagged.
        """
        X_new    = [101, 99, 350, 98, 100]
        detector = ZScoreDetector(threshold = 3.0)
        detector.fit(clean_1d)
        result = detector.predict(X_new)
        assert result[2] is np.bool_(True), \
            "Value 350 at index 2 should be flagged as anomaly."

    def test_predict_1d_output_shape(self, clean_1d):
        """
        Purpose  : verify that predict() output shape matches input length.
        Expected : shape == (5,).
        Obtained : boolean array of same length as X_new.
        """
        X_new    = [101, 99, 105, 98, 100]
        detector = ZScoreDetector()
        detector.fit(clean_1d)
        result = detector.predict(X_new)
        assert result.shape == (len(X_new),)

    def test_predict_1d_returns_zscores(self, clean_1d):
        """
        Purpose  : verify that return_zscore = True returns a tuple
                   (anomalies, z_scores) with matching shapes for 1D.
        Expected : both arrays have shape (5,).
        Obtained : tuple of two numpy arrays of the same length.
        """
        X_new    = [101, 99, 350, 98, 100]
        detector = ZScoreDetector()
        detector.fit(clean_1d)
        anomalies, zscores = detector.predict(X_new, return_zscore = True)
        assert anomalies.shape == zscores.shape
        assert len(zscores) == len(X_new)

    # 2.5  predict() — 2D

    def test_predict_2d_axis0_no_anomaly(self, clean_2d):
        """
        Purpose  : verify that predict() returns all False on clean 2D data
                   when fitted on clean 2D training data with axis = 0.
        Expected : all False, shape == (5, 3).
        Obtained : all Z-scores well below threshold=3.0.
        """
        detector = ZScoreDetector(threshold = 3.0, axis = 0)
        detector.fit(clean_2d)
        result = detector.predict(clean_2d)
        assert result.shape == clean_2d.shape
        assert not np.any(result)

    def test_predict_2d_axis0_detects_anomaly(self, clean_2d, anomalous_2d):
        """
        Purpose  : verify that fit() on clean data + predict() on anomalous
                   data correctly flags the outlier row (index 2) with axis=0.
        Expected : at least one True in row 2 of the result.
        Obtained : extreme values in row 2 produce Z-scores >> threshold=2.0.
        """
        detector = ZScoreDetector(threshold = 2.0, axis = 0)
        detector.fit(clean_2d)
        result = detector.predict(anomalous_2d)
        assert result.shape == anomalous_2d.shape
        assert np.any(result[2]), \
            "Row 2 with extreme values should be flagged as anomalous."

    def test_predict_2d_axis1_output_shape(self, clean_2d):
        """
        Purpose  : verify output shape for 2D predict() with axis=1.
        Expected : shape == (5, 3).
        Obtained : output shape matches input shape.
        """
        detector = ZScoreDetector(axis = 1)
        detector.fit(clean_2d)
        result = detector.predict(clean_2d)
        assert result.shape == clean_2d.shape

    def test_predict_2d_returns_zscores(self, clean_2d):
        """
        Purpose  : verify that return_zscore=True works correctly for 2D predict().
        Expected : both arrays have shape (5, 3).
        Obtained : tuple of two numpy arrays matching input shape.
        """
        detector = ZScoreDetector(axis=0)
        detector.fit(clean_2d)
        anomalies, zscores = detector.predict(clean_2d, return_zscore=True)
        assert anomalies.shape == clean_2d.shape
        assert zscores.shape   == clean_2d.shape

    # 2.6  Error handling

    def test_predict_before_fit_raises_runtimeerror(self):
        """
        Purpose  : verify that calling predict() before fit() raises RuntimeError.
        Expected : RuntimeError with an explicit message.
        Obtained : RuntimeError because is_fitted_ = False.
        """
        detector = ZScoreDetector()
        with pytest.raises(RuntimeError):
            detector.predict([100, 102, 98, 101, 99])

    def test_fit_too_few_points_raises_valueerror(self):
        """
        Purpose  : verify that fitting with fewer than MIN_DATA_POINTS raises ValueError.
        Expected : ValueError.
        Obtained : ValueError because only 3 points provided.
        """
        detector = ZScoreDetector()
        with pytest.raises(ValueError):
            detector.fit([1, 2, 3])

    def test_fit_constant_data_raises_valueerror(self):
        """
        Purpose  : verify that fitting on constant data (std = 0) raises ValueError
                   because the Z-score is mathematically undefined.
        Expected : ValueError.
        Obtained : ValueError because std = 0 → division by zero.
        """
        detector = ZScoreDetector()
        with pytest.raises(ValueError):
            detector.fit([5, 5, 5, 5, 5, 5, 5])

    def test_fit_invalid_type_raises_typeerror(self):
        """
        Purpose  : verify that passing an invalid type to fit() raises TypeError.
        Expected : TypeError.
        Obtained : TypeError because input is a string.
        """
        detector = ZScoreDetector()
        with pytest.raises(TypeError):
            detector.fit("invalid input")

    def test_predict_invalid_type_raises_typeerror(self, clean_1d):
        """
        Purpose  : verify that passing an invalid type to predict() raises TypeError.
        Expected : TypeError.
        Obtained : TypeError because input is a dict.
        """
        detector = ZScoreDetector()
        detector.fit(clean_1d)
        with pytest.raises(TypeError):
            detector.predict({"key": "value"})
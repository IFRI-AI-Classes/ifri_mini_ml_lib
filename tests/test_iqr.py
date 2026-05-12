"""
Unit tests for the IQR anomaly detection module.

Tests cover:
    - Initialization (default and custom factor)
    - fit() computation of Q1, Q3, IQR, bounds
    - predict() anomaly labeling on 1D and 2D data
    - fit_predict() convenience method
    - get_bounds() accessor
    - summary() console output
    - Error handling (predict before fit, empty data, feature mismatch)
    - Robustness: NaN handling, constant features, single sample.
"""

import numpy as np
import pytest
import warnings

from ifri_mini_ml_lib.anomalies_detection.iqr import IQR


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def simple_1d_data():
    """1D dataset with a clear outlier at 100."""
    return np.array([[1], [2], [3], [4], [5], [100]])


@pytest.fixture
def multivariate_data():
    """2D dataset: the last sample has an outlier on the first feature."""
    return np.array([
        [1, 10],
        [2, 12],
        [3, 11],
        [4, 13],
        [5, 12],
        [100, 11],  # anomaly on feature 0
    ])


@pytest.fixture
def normal_data():
    """Dataset with no outliers."""
    return np.array([[2], [3], [4], [5], [6]])


# ─────────────────────────────────────────────
# Test: Initialization
# ─────────────────────────────────────────────

def test_init_default():
    """Default factor should be 1.5."""
    detector = IQR()
    assert detector.factor == 1.5
    assert detector._is_fitted is False


def test_init_custom_factor():
    """Custom factor should be stored correctly."""
    detector = IQR(factor=3.0)
    assert detector.factor == 3.0


def test_init_invalid_factor():
    """Negative or zero factor should raise ValueError."""
    with pytest.raises(ValueError):
        IQR(factor=0)
    with pytest.raises(ValueError):
        IQR(factor=-1.0)


# ─────────────────────────────────────────────
# Test: fit()
# ─────────────────────────────────────────────

def test_fit_computes_bounds(simple_1d_data):
    """After fit(), Q1, Q3, IQR, bounds should be set."""
    detector = IQR(factor=1.5)
    result = detector.fit(simple_1d_data)

    # fit() should return self for method chaining
    assert result is detector
    assert detector._is_fitted is True
    assert detector.Q1_ is not None
    assert detector.Q3_ is not None
    assert detector.IQR_ is not None
    assert detector.lower_bound_ is not None
    assert detector.upper_bound_ is not None


def test_fit_correct_quartiles():
    """Verify Q1, Q3, IQR on a known dataset."""
    # Data: [1, 2, 3, 4, 5, 6, 7, 8]
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    detector = IQR(factor=1.5)
    detector.fit(X)

    # numpy percentile with linear interpolation:
    # Q1 = 2.75, Q3 = 6.25, IQR = 3.5
    assert detector.Q1_[0] == pytest.approx(2.75)
    assert detector.Q3_[0] == pytest.approx(6.25)
    assert detector.IQR_[0] == pytest.approx(3.5)
    assert detector.lower_bound_[0] == pytest.approx(2.75 - 1.5 * 3.5)
    assert detector.upper_bound_[0] == pytest.approx(6.25 + 1.5 * 3.5)


def test_fit_empty_data():
    """Fitting on empty data should raise ValueError."""
    detector = IQR()
    with pytest.raises(ValueError):
        detector.fit([])


def test_fit_1d_flat_array():
    """fit() should accept a flat 1D list and reshape internally."""
    detector = IQR()
    detector.fit([1, 2, 3, 4, 5])
    assert detector._is_fitted is True
    assert detector.Q1_.shape == (1,)


# ─────────────────────────────────────────────
# Test: predict()
# ─────────────────────────────────────────────

def test_predict_detects_outlier(simple_1d_data):
    """The value 100 should be flagged as an anomaly."""
    detector = IQR(factor=1.5)
    detector.fit(simple_1d_data)
    labels = detector.predict(simple_1d_data)

    assert labels.shape == (6,)
    # 100 is the last sample → should be 1 (anomaly)
    assert labels[-1] == 1
    # The first 5 normal values should be 0
    assert all(labels[:-1] == 0)


def test_predict_no_outliers(normal_data):
    """A well-behaved dataset should produce all zeros."""
    detector = IQR(factor=1.5)
    detector.fit(normal_data)
    labels = detector.predict(normal_data)

    assert np.all(labels == 0)


def test_predict_multivariate(multivariate_data):
    """Anomaly on one feature should flag the entire sample."""
    detector = IQR(factor=1.5)
    detector.fit(multivariate_data)
    labels = detector.predict(multivariate_data)

    # Last sample [100, 11] should be anomalous (feature 0 is outlier)
    assert labels[-1] == 1


def test_predict_before_fit():
    """Calling predict() before fit() should raise RuntimeError."""
    detector = IQR()
    with pytest.raises(RuntimeError):
        detector.predict([[1], [2]])


def test_predict_feature_mismatch():
    """Mismatched feature count between fit and predict should raise."""
    detector = IQR()
    detector.fit([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        detector.predict([[1], [2]])  # 1 feature instead of 2


# ─────────────────────────────────────────────
# Test: Robustness
# ─────────────────────────────────────────────

def test_handle_nan_raise():
    """Should raise ValueError if NaNs are found with handle_nan='raise'."""
    detector = IQR(handle_nan='raise')
    X = [[1], [np.nan], [3]]
    with pytest.raises(ValueError, match="contains NaN values"):
        detector.fit(X)

def test_handle_nan_omit():
    """Should ignore NaNs with handle_nan='omit'."""
    detector = IQR(handle_nan='omit')
    X = [[1], [np.nan], [2], [3], [4]]
    # Calculations should ignore NaN
    detector.fit(X)
    assert detector._is_fitted
    # Q1 of [1, 2, 3, 4] is 1.75
    assert detector.Q1_[0] == pytest.approx(1.75)

def test_constant_feature_warning():
    """Should issue a UserWarning for constant features."""
    detector = IQR()
    X = [[5], [5], [5], [5]]
    with pytest.warns(UserWarning, match="constant"):
        detector.fit(X)

def test_single_sample_raises():
    """Should raise ValueError for datasets with fewer than 2 samples."""
    detector = IQR()
    with pytest.raises(ValueError, match="At least 2 samples"):
        detector.fit([[1]])

def test_non_numeric_data_raises():
    """Should raise TypeError for non-numeric data."""
    detector = IQR()
    with pytest.raises(ValueError): # np.array(..., dtype=float) raises ValueError for non-numeric strings
        detector.fit([["a"], ["b"]])

# ─────────────────────────────────────────────
# Other Methods
# ─────────────────────────────────────────────

def test_fit_predict(simple_1d_data):
    """fit_predict() should return the same labels as fit() then predict()."""
    detector1 = IQR(factor=1.5)
    detector1.fit(simple_1d_data)
    labels_separate = detector1.predict(simple_1d_data)

    detector2 = IQR(factor=1.5)
    labels_combined = detector2.fit_predict(simple_1d_data)

    np.testing.assert_array_equal(labels_separate, labels_combined)


def test_get_bounds(simple_1d_data):
    """get_bounds() should return a dict with 'lower' and 'upper' keys."""
    detector = IQR()
    detector.fit(simple_1d_data)
    bounds = detector.get_bounds()

    assert "lower" in bounds
    assert "upper" in bounds


def test_summary_runs_without_error(simple_1d_data, capsys):
    """summary() should print output without raising."""
    detector = IQR()
    detector.fit(simple_1d_data)
    detector.summary()

    captured = capsys.readouterr()
    assert "IQR Detector Summary" in captured.out

"""Unit tests for Z-Score Anomaly Detection"""
import sys
import os
import pytest
import numpy as np

# ============================================================================
# IMPORT MANAGEMENT (compatible with project structure)
# ============================================================================

def import_z_score():
    """Attempts to import z_score from different possible paths"""
    
    # List of possible paths
    """Checks multiple paths to import z_score, ensuring compatibility with different project structures.
    This function tries to import the z_score module from various locations, allowing the tests to run"""

    possible_paths = [
        # Path 1: from root with ifri_mini_ml_lib
        os.path.join(os.path.dirname(__file__), '..', 'ifri_mini_ml_lib', 'anomalies_detection'),
        # Path 2: from root with ifri_ml_mini
        os.path.join(os.path.dirname(__file__), '..', 'ifri_ml_mini', 'anomalies_detection'),
        # Path 3: parent directory
        os.path.join(os.path.dirname(__file__), '..'),
        # Path 4: current directory
        os.path.dirname(__file__),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            try:
                #  IMPORT CORRECT : fonctions, pas une classe
                from z_score import (
                    zscore_detection,
                    modified_zscore_detection
                )
                print(f"Import successful from: {path}")
                return zscore_detection, modified_zscore_detection
            except ImportError:
                continue
    
    raise ImportError("Unable to import z_score functions. Please check the paths.")

# Retrieve the functions
zscore_detection, modified_zscore_detection = import_z_score()


# ============================================================================
# TESTS
# ============================================================================

class TestZScoreDetection:
    """Tests for zscore_detection function"""
    
    def test_basic_anomaly_detection(self):
        """Basic test: detect a single anomaly"""
        data = [100, 102, 98, 101, 99, 300, 101, 98]
        # Use a lower threshold to ensure detection of the anomaly at index 5
        anomalies = zscore_detection(data, threshold=2.5)

        assert anomalies[5] == True
        assert sum(anomalies) == 1
    
    def test_no_anomaly(self):
        """Test: no anomaly in normal data"""
        data = [100, 101, 99, 100, 102, 98, 101, 99]
        anomalies = zscore_detection(data, threshold=3.0)
        assert not any(anomalies)
    
    def test_returns_z_scores(self):
        """Test: returns Z-scores when requested"""
        data = [100, 102, 98, 101, 99, 300, 101, 98]
        anomalies, z_scores = zscore_detection(data, return_zscore=True)
        assert isinstance(z_scores, np.ndarray)
        assert len(z_scores) == len(data)
    
    def test_constant_data_raises_error(self):
        """Test: constant data raises an error"""
        data = [42, 42, 42, 42, 42]
        with pytest.raises(ValueError, match="Standard deviation is zero"):
            zscore_detection(data)
    
    def test_negative_values_work(self):
        """Test: works with negative values"""
        data = [-100, -102, -98, -101, -99, -500, -101, -98]
        anomalies = zscore_detection(data, threshold=2.5)
        assert anomalies[5] == True
    
    def test_float_values_work(self):
        """Test: works with float values"""
        data = [10.5, 10.6, 10.4, 10.7, 10.5, 200.0, 10.6, 10.4]
        anomalies = zscore_detection(data, threshold=2.5)
        assert anomalies[5] == True
    
    def test_numpy_array_input(self):
        """Test: accepts numpy array as input"""
        data = np.array([100, 102, 98, 101, 99, 300, 101, 98])
        anomalies = zscore_detection(data, threshold=2.5)
        assert anomalies[5] == True


class TestModifiedZScoreDetection:
    """Tests for modified_zscore_detection function"""
    
    def test_modified_detects_anomaly(self):
        """Test: Modified Z-score detects anomalies"""
        data = [10, 12, 11, 10, 1000, 11, 12, 10]
        anomalies = modified_zscore_detection(data, threshold=3.5)
        assert sum(anomalies) >= 1
    
    def test_modified_returns_scores(self):
        """Test: Modified Z-score returns scores when requested"""
        data = [10, 12, 11, 10, 1000, 11, 12, 10]
        anomalies, mz_scores = modified_zscore_detection(data, return_zscore=True)
        assert isinstance(mz_scores, np.ndarray)
        assert len(mz_scores) == len(data)
    
    def test_modified_robust_to_multiple_anomalies(self):
        """Test: Modified Z-score is robust with multiple anomalies"""
        data = [10, 12, 11, 10, 1000, 11, 12, 10, 2000, 11]
        anomalies = modified_zscore_detection(data, threshold=3.5)
        assert sum(anomalies) >= 2


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_empty_data_raises_error(self):
        """Test: empty data raises error"""
        with pytest.raises(ValueError, match="at least.*points"):
            zscore_detection([])
    
    def test_too_few_points_raises_error(self):
        """Test: too few data points raises error"""
        with pytest.raises(ValueError, match="at least.*points"):
            zscore_detection([1, 2, 3, 4])
    
    def test_invalid_threshold_raises_error(self):
        """Test: invalid threshold raises error"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        with pytest.raises(ValueError, match="positive number"):
            zscore_detection(data, threshold=-1.0)
        with pytest.raises(ValueError, match="positive number"):
            zscore_detection(data, threshold=0)
    
    def test_invalid_data_type_raises_error(self):
        """Test: invalid data type raises error"""
        with pytest.raises(TypeError, match="must be a list or numpy.ndarray"):
            zscore_detection("not a list")


# ============================================================================
# DIRECT EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING Z-SCORE TESTS")
    print("=" * 60)
    
    # Quick test
    data = [100, 102, 98, 101, 99, 300, 101, 98]
    anomalies = zscore_detection(data, threshold=3.0)
    
    print(f"Data: {data}")
    print(f"Anomalies: {anomalies}")
    print(f"Number of anomalies: {sum(anomalies)}")
    
    if anomalies[5]:
        print("\n BASIC TEST PASSED!")
    else:
        print("\nBASIC TEST FAILED")
    
    print("\n" + "=" * 60)
    print("To run all tests: pytest test_z_score.py -v")
    print("=" * 60)
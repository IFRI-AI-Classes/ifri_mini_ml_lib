import numpy as np
import pytest
from ifri_mini_ml_lib.metrics.bias_evaluation import (
    selection_rate,
    selection_rate_per_group,
    demographic_parity_ratio,
    demographic_parity_difference,
    tpr_fpr_by_group,
    equalized_odds_ratio,
    equalized_odds_difference,
)

# Test selection_rate
def test_selection_rate_basic():
    y_pred = np.array([1, 0, 1, 1])
    assert selection_rate(None, y_pred, pos_label=1) == 0.75

def test_selection_rate_with_weights():
    y_pred = np.array([1, 0, 1, 1])
    weights = np.array([0.5, 0.5, 1.0, 1.0])
    assert selection_rate(None, y_pred, pos_label=1, sample_weight=weights) == 2.5 / 3.0

def test_selection_rate_empty_predictions():
    with pytest.raises(ValueError):
        selection_rate(None, np.array([]), pos_label=1)

def test_selection_rate_different_pos_label():
    y_pred = np.array([2, 0, 2, 2])
    assert selection_rate(None, y_pred, pos_label=2) == 0.75

# Test selection_rate_per_group
def test_selection_rate_per_group_basic():
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    rates = selection_rate_per_group(None, y_pred, sensitive, pos_label=1)
    assert rates['A'] == 0.5
    assert rates['B'] == 0.0

def test_selection_rate_per_group_with_weights():
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    rates = selection_rate_per_group(None, y_pred, sensitive, pos_label=1, sample_weight=weights)
    assert rates['A'] == 4.0 / 4.0
    assert rates['B'] == 0.0

# Test demographic_parity_ratio
def test_demographic_parity_ratio_perfect():
    y_pred = np.array([1, 1, 1, 1])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    ratio, rates = demographic_parity_ratio(y_pred, sensitive, pos_label=1)
    assert ratio == 1.0
    assert rates['A'] == 1.0
    assert rates['B'] == 1.0

def test_demographic_parity_ratio_zero():
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    ratio, rates = demographic_parity_ratio(y_pred, sensitive, pos_label=1)
    assert ratio == 0.0
    assert rates['A'] == 1.0
    assert rates['B'] == 0.0

# Test demographic_parity_difference
def test_demographic_parity_difference_perfect():
    y_pred = np.array([1, 1, 1, 1])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    diff, rates = demographic_parity_difference(y_pred, sensitive, pos_label=1)
    assert diff == 0.0
    assert rates['A'] == 1.0
    assert rates['B'] == 1.0

def test_demographic_parity_difference_max():
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    diff, rates = demographic_parity_difference(y_pred, sensitive, pos_label=1)
    assert diff == 1.0
    assert rates['A'] == 1.0
    assert rates['B'] == 0.0

# Test tpr_fpr_by_group
def test_tpr_fpr_by_group_basic():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    tpr, fpr = tpr_fpr_by_group(y_true, y_pred, sensitive, pos_label=1)
    assert tpr['A'] == 1.0
    assert tpr['B'] == 0.0
    assert fpr['A'] == 1.0
    assert fpr['B'] == 0.0

def test_tpr_fpr_by_group_no_positives():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    tpr, fpr = tpr_fpr_by_group(y_true, y_pred, sensitive, pos_label=1)
    assert tpr['A'] == 0.0
    assert tpr['B'] == 0.0
    assert fpr['A'] == 1.0
    assert fpr['B'] == 0.0

# Test equalized_odds_ratio
def test_equalized_odds_ratio_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    ratio, tpr, fpr = equalized_odds_ratio(y_true, y_pred, sensitive, pos_label=1)
    assert ratio == 1.0
    assert tpr['A'] == 1.0
    assert tpr['B'] == 1.0
    assert fpr['A'] == 0.0
    assert fpr['B'] == 0.0

def test_equalized_odds_ratio_zero():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    ratio, tpr, fpr = equalized_odds_ratio(y_true, y_pred, sensitive, pos_label=1)
    assert ratio == 0.0
    assert tpr['A'] == 1.0
    assert tpr['B'] == 0.0
    assert fpr['A'] == 1.0
    assert fpr['B'] == 0.0

# Test equalized_odds_difference
def test_equalized_odds_difference_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    diff, tpr, fpr = equalized_odds_difference(y_true, y_pred, sensitive, pos_label=1)
    assert diff == 0.0
    assert tpr['A'] == 1.0
    assert tpr['B'] == 1.0
    assert fpr['A'] == 0.0
    assert fpr['B'] == 0.0

def test_equalized_odds_difference_max():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array(['A', 'B', 'A', 'B'])
    diff, tpr, fpr = equalized_odds_difference(y_true, y_pred, sensitive, pos_label=1)
    assert diff == 1.0
    assert tpr['A'] == 1.0
    assert tpr['B'] == 0.0
    assert fpr['A'] == 1.0
    assert fpr['B'] == 0.0
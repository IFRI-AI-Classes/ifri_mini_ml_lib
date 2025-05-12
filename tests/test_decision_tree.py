import numpy as np
import pytest
from ifri_mini_ml_lib.classification.decision_tree import DecisionTree

def test_decision_tree_basic_split():
    X = np.array([
        [1, 1],
        [2, 1],
        [3, 1],
        [10, 2],
        [11, 2],
        [12, 2]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    
    preds = tree.predict(X)
    assert np.array_equal(preds, y)

def test_decision_tree_min_samples_split():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    tree = DecisionTree(min_samples_split=4)
    tree.fit(X, y)

    # All samples should be assigned the majority class
    preds = tree.predict(X)
    assert np.all(preds == 0)

def test_predict_raises_error_if_not_fit():
    tree = DecisionTree()
    X = np.array([[1], [2]])
    with pytest.raises(ValueError):
        tree.predict(X)

def test_empty_dataset_handling():
    X = np.empty((0, 2))
    y = np.array([])
    tree = DecisionTree()
    result = tree.fit(X, y)
    assert result == 0

def test_entropy_and_most_common_label():
    tree = DecisionTree()
    y = np.array([0, 0, 1, 1, 1])
    entropy = tree._entropy(y)
    assert 0 < entropy < 1  # Entropy should be between 0 and 1
    assert tree._most_common_label(y) == 1

def test_information_gain_zero_when_split_invalid():
    tree = DecisionTree()
    y = np.array([0, 1])
    left_mask = np.array([True, False])
    right_mask = ~left_mask
    gain = tree._information_gain(y, np.array([False, False]), np.array([True, True]), 1.0)
    assert gain == 0.0

import numpy as np
import pytest
from ifri_mini_ml_lib.metrics.evaluation_metrics import *

# Fixtures pour les données de test
@pytest.fixture
def regression_data_perfect():
    y_true = [3.0, 5.0, 7.0]
    y_pred = [3.0, 5.0, 7.0]
    return y_true, y_pred

@pytest.fixture
def regression_data_errors():
    y_true = [2.0, 4.0, 6.0]
    y_pred = [3.0, 5.0, 5.0]
    return y_true, y_pred

@pytest.fixture
def regression_data_zero_division():
    y_true = [0.0, 2.0, 4.0]
    y_pred = [0.0, 3.0, 4.0]
    return y_true, y_pred

@pytest.fixture
def classification_data_simple():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    return y_true, y_pred

@pytest.fixture
def clustering_data_perfect():
    data = np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 6.0]])
    labels = np.array([0, 0, 1])
    centroids = np.array([[1.0, 2.0], [5.0, 6.0]])
    return data, labels, centroids

@pytest.fixture
def clustering_data_silhouette():
    data = np.array([[0.0], [0.0], [1.0], [1.0]])
    labels = np.array([0, 0, 1, 1])
    return data, labels

# Tests pour evaluate_model
def test_evaluate_model_perfect_predictions(regression_data_perfect):
    
    y_true, y_pred = regression_data_perfect
    metrics = evaluate_model(y_true, y_pred)
    
    assert metrics["MSE"] == pytest.approx(0.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["MAPE"] == pytest.approx(0.0)
    assert metrics["R²"] == pytest.approx(1.0)

def test_evaluate_model_with_errors(regression_data_errors):
    
    y_true, y_pred = regression_data_errors
    metrics = evaluate_model(y_true, y_pred)
    
    assert metrics["MSE"] == pytest.approx(1.0)
    assert metrics["RMSE"] == pytest.approx(1.0)
    assert metrics["MAE"] == pytest.approx(1.0)
    assert metrics["MAPE"] == pytest.approx(30.55555555555, rel=1e-3)
    assert metrics["R²"] == pytest.approx(0.625)

def test_evaluate_model_zero_division(regression_data_zero_division):
    
    y_true, y_pred = regression_data_zero_division
    metrics = evaluate_model(y_true, y_pred)
    
    assert metrics["MAPE"] == pytest.approx(16.666666666666668)

def test_evaluate_model_length_mismatch():
    
    with pytest.raises(ValueError):
        evaluate_model([1, 2], [3])

# Tests pour confusion_matrix
def test_confusion_matrix_simple(classification_data_simple):
    
    y_true, y_pred = classification_data_simple
    matrix = confusion_matrix(y_true, y_pred)
    
    assert matrix[0][0] == 1
    assert matrix[0][1] == 1
    assert matrix[1][0] == 1
    assert matrix[1][1] == 1

# Tests pour accuracy
def test_accuracy(classification_data_simple):
    
    y_true, y_pred = classification_data_simple
    assert accuracy(y_true, y_pred) == pytest.approx(0.5)

# Tests pour precision
def test_precision(classification_data_simple):
    
    y_true, y_pred = classification_data_simple
    assert precision(y_true, y_pred, positive_class=1) == pytest.approx(0.5)

# Tests pour recall
def test_recall(classification_data_simple):
    
    y_true, y_pred = classification_data_simple
    assert recall(y_true, y_pred, positive_class=1) == pytest.approx(0.5)

# Tests pour f1_score
def test_f1_score(classification_data_simple):
    
    y_true, y_pred = classification_data_simple
    assert f1_score(y_true, y_pred, positive_class=1) == pytest.approx(0.5)

# Tests pour calculate_inertia
def test_calculate_inertia_perfect(clustering_data_perfect):
    
    data, labels, centroids = clustering_data_perfect
    assert calculate_inertia(data, labels, centroids) == pytest.approx(0.0)

# Tests pour calculate_silhouette
"""def test_calculate_silhouette_perfect(clustering_data_silhouette):
    
    data, labels = clustering_data_silhouette
    assert calculate_silhouette(data, labels) == pytest.approx(1.0)
"""
def test_calculate_silhouette_single_cluster():
    
    data = np.array([[1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 0])
    assert calculate_silhouette(data, labels) == pytest.approx(0.0)
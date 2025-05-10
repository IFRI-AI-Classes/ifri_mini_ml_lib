import pytest
import numpy as np
from ifri_mini_ml_lib.classification.knn import KNN  

def test_knn_classification_simple():
    X = [[0], [1], [2], [3]]
    y = ['A', 'A', 'B', 'B']
    knn = KNN(k=3, task='classification')
    knn.fit(X, y)
    assert knn.predict([[1.5]]) == ['A']

def test_knn_regression_simple():
    X = [[0], [1], [2], [3]]
    y = [1.0, 2.0, 3.0, 4.0]
    knn = KNN(k=2, task='regression')
    knn.fit(X, y)
    prediction = knn.predict([[2.5]])[0]
    assert prediction == pytest.approx((3.0 + 4.0) / 2)

def test_knn_k_greater_than_data_length():
    X = [[0], [1]]
    y = ['A', 'B']
    knn = KNN(k=5, task='classification')
    knn.fit(X, y)
    result = knn.predict([[0.5]])
    assert result[0] in ['A', 'B']  

def test_knn_not_fitted_raises_error():
    knn = KNN()
    with pytest.raises(AttributeError):
        knn.predict([[1]])

def test_knn_classification_tie_break():
    # Deux voisins 'A', deux voisins 'B', tie → 'A' car 'A' < 'B'
    X = [[0], [1], [2], [3]]
    y = ['A', 'A', 'B', 'B']
    knn = KNN(k=4, task='classification')
    knn.fit(X, y)
    assert knn.predict([[1.5]]) == ['A']

def test_knn_multiple_predictions():
    X = [[0], [1], [2], [3]]
    y = ['A', 'A', 'B', 'B']
    knn = KNN(k=1, task='classification')
    knn.fit(X, y)
    preds = knn.predict([[0], [3]])
    assert preds == ['A', 'B']

def test_knn_regression_float_precision():
    X = [[1], [2], [3]]
    y = [2.0, 4.0, 6.0]
    knn = KNN(k=3, task='regression')
    knn.fit(X, y)
    prediction = knn.predict([[2]])[0]
    assert prediction == pytest.approx(4.0)

import pytest
import numpy as np
from ifri_mini_ml_lib.classification.naive_bayes import NaiveBayes

@pytest.fixture
def data():
    X = np.array([[1, 100], [2, 200], [10, 1000], [11, 1100]])
    y = np.array([0, 0, 1, 1])
    return X, y

def test_fit_shapes(data):
    """Vérifie que le modèle stocke les bons paramètres après le fit."""
    X, y = data
    nb = NaiveBayes()
    nb.fit(X, y)
    
    assert len(nb.classes) == 2
    assert nb.scaler_mean.shape == (2,)
    assert len(nb.mean) == 2

def test_standardization_logic(data):
    """Vérifie que la standardisation interne fonctionne (moyenne proche de 0)."""
    X, y = data
    nb = NaiveBayes()
    nb.fit(X, y)
    
    
    for c in nb.mean:
        assert np.all(nb.mean[c] < 5) 
        assert np.all(nb.mean[c] > -5)

def test_predict_output_format(data):
    """Vérifie que predict renvoie le bon nombre d'éléments."""
    X, y = data
    nb = NaiveBayes()
    nb.fit(X, y)
    preds = nb.predict(X)
    
    assert isinstance(preds, list)
    assert len(preds) == len(y)

def test_perfect_separation():
    """Vérifie que le modèle classifie parfaitement un cas simple."""
    X = np.array([[1], [2], [10], [11]])
    y = np.array([0, 0, 1, 1])
    nb = NaiveBayes()
    nb.fit(X, y)
    preds = nb.predict(np.array([[1.5], [10.5]]))
    
    assert preds == [0, 1]
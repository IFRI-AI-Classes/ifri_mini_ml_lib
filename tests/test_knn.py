import numpy as np
import pytest
from ifri_mini_ml_lib.classification.knn import KNN

@pytest.fixture
def binary_data():
    """Données linéairement séparables en 2D"""
    X = np.array([[1, 2], [1.5, 2.5], [2, 1], [5, 6], [5.5, 6.5]])
    y = np.array([0, 0, 0, 1, 1])
    return X, y

@pytest.fixture
def multi_data():
    """Données multiclasses en 3D"""
    X = np.array([[1, 2, 3], [1.1, 2.1, 3.1], [4, 5, 6], [4.1, 5.1, 6.1], [7, 8, 9]])
    y = np.array([0, 0, 1, 1, 2])
    return X, y

@pytest.fixture
def regression_data():
    """Données pour régression linéaire simple"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([10, 20, 30, 40])
    return X, y

class TestKNN:
    def test_initialization(self):
        """Teste l'initialisation avec différents paramètres"""
        for k in [1, 3, 5]:
            for task in ['classification', 'regression']:
                model = KNN(k=k, task=task)
                assert model.k == k
                assert model.task == task

    def test_binary_classification(self, binary_data):
        """Teste la classification binaire"""
        X, y = binary_data
        model = KNN(k=3)
        model.fit(X, y)
        
        # Test sur les points d'entraînement
        preds = model.predict(X)
        assert np.array_equal(preds, y)
        
        # Test sur nouveaux points
        test_points = np.array([[1.2, 2.3], [5.2, 6.2]])
        assert np.array_equal(model.predict(test_points), [0, 1])

    def test_multiclass_classification(self, multi_data):
        """Teste la classification multiclasse"""
        X, y = multi_data
        model = KNN(k=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.array_equal(preds, y)

    def test_regression(self, regression_data):
        """Teste le mode régression"""
        X, y = regression_data
        model = KNN(k=2, task='regression')
        model.fit(X, y)
        
        # Test interpolation
        assert np.isclose(model.predict(np.array([[2.5]]))[0], 25.0)
        
        # Test extrapolation
        assert np.isclose(model.predict(np.array([[0]]))[0], 15.0)

    def test_edge_cases(self):
        """Teste les cas limites"""
        # k > n_samples
        model = KNN(k=10)
        model.fit(np.array([[1], [2]]), np.array([10, 20]))
        assert 10 <= model.predict(np.array([[1.5]]))[0] <= 20
        
        # Tous points identiques
        model.fit(np.array([[1], [1], [1]]), np.array([0, 0, 1]))
        assert model.predict(np.array([[1]]))[0] in {0, 1}
        
        # Données haute dimension
        model.fit(np.random.rand(10, 100), np.array([0]*5 + [1]*5))
        assert model.predict(np.random.rand(1, 100))[0] in {0, 1}

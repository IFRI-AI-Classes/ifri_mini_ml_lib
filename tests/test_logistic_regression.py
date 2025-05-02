import numpy as np
import pytest
from ifri_mini_lib.classification.logistic_regression import LogisticRegression

@pytest.fixture
def binary_data():
    """Données linéairement séparables"""
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 3]])
    y = np.array([0, 0, 1, 1])
    return X, y

@pytest.fixture
def real_data():
    """Charge le dataset breast cancer"""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data[:, :5]  # Premières 5 features pour la vitesse
    y = data.target
    return X, y

@pytest.fixture
def non_linear_data():
    """Données non linéairement séparables"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y

class TestLogisticRegression:
    def test_initialization(self):
        """Teste différents paramètres initiaux"""
        for lr in [0.001, 0.01, 0.1]:
            for max_iter in [100, 1000]:
                model = LogisticRegression(learning_rate=lr, max_iter=max_iter)
                assert model.learning_rate == lr
                assert model.max_iter == max_iter

    def test_binary_classification(self, binary_data):
        """Teste sur données séparables linéairement"""
        X, y = binary_data
        model = LogisticRegression(learning_rate=0.1, max_iter=1000)
        model.fit(X, y)
        
        # Vérifie la décroissance de la loss
        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        
        # Vérifie l'accuracy
        preds = model.predict(X)
        assert np.mean(preds == y) == 1.0

    def test_real_data(self, real_data):
        """Teste sur données réelles"""
        X, y = real_data
        model = LogisticRegression(max_iter=2000)
        model.fit(X, y)
        assert np.mean(model.predict(X) == y) > 0.9

    def test_non_linear_data(self, non_linear_data):
        """Teste sur données non linéaires"""
        X, y = non_linear_data
        model = LogisticRegression(max_iter=3000)
        model.fit(X, y)
        assert np.mean(model.predict(X) == y) > 0.7

    def test_probability_output(self, binary_data):
        """Teste le output des probabilités"""
        X, y = binary_data
        model = LogisticRegression()
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert np.all((probas >= 0) & (probas <= 1))
        assert probas.shape == y.shape

    def test_edge_cases(self):
        """Teste les cas limites"""
        # Une seule classe
        model = LogisticRegression()
        model.fit(np.random.rand(10, 2), np.zeros(10))
        assert np.all(model.predict(np.random.rand(5, 2)) == 0)
        
        # Données non normalisées
        X = np.array([[1000], [2000]])
        y = np.array([0, 1])
        model.fit(X, y)
        assert model.predict(np.array([[1500]]))[0] == 1
        
        # Grand nombre de features
        model.fit(np.random.rand(10, 100), np.random.randint(0, 2, 10))
        assert model.predict(np.random.rand(1, 100))[0] in {0, 1}

import numpy as np
import pytest
from ifri_mini_ml_lib.classification.logistic_regression import LogisticRegression

@pytest.fixture
def binary_data():
    """Données linéairement séparables simples"""
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [4.0, 3.0],
        [1.5, 2.5],
        [3.5, 2.5]
    ])
    y = np.array([0, 0, 1, 1, 0, 1])
    return X, y

@pytest.fixture
def real_data():
    """Données simulées plus complexes"""
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 5)  # 5 features
    
    # Création d'une relation linéaire + bruit
    coefficients = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    linear_combination = np.dot(X, coefficients) + 0.5 * np.random.randn(n_samples)
    y_proba = 1 / (1 + np.exp(-linear_combination))
    y = (y_proba > 0.5).astype(int)
    
    return X, y

@pytest.fixture 
def non_linear_data():
    """Données avec relation non-linéaire (XOR)"""
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)  # Relation XOR
    return X, y

@pytest.fixture
def edge_case_data():
    """Cas limites"""
    # Une seule classe
    X_single = np.random.rand(20, 3)
    y_single = np.zeros(20)
    
    # Données non normalisées
    X_large = np.array([[1000], [2000], [1500]])
    y_large = np.array([0, 1, 0])
    
    # Beaucoup de features
    X_high_dim = np.random.rand(10, 100)
    y_high_dim = np.random.randint(0, 2, 10)
    
    return {
        'single_class': (X_single, y_single),
        'large_values': (X_large, y_large),
        'high_dim': (X_high_dim, y_high_dim)
    }

class TestLogisticRegression:
    def test_initialization(self):
        """Teste l'initialisation avec différents paramètres"""
        for lr in [0.001, 0.01, 0.1]:
            for max_iter in [100, 1000]:
                model = LogisticRegression(learning_rate=lr, max_iter=max_iter)
                assert model.learning_rate == lr
                assert model.max_iter == max_iter
                assert model.weights is None
                assert model.bias is None

    def test_binary_classification(self, binary_data):
        """Teste sur données linéairement séparables"""
        X, y = binary_data
        model = LogisticRegression(learning_rate=0.1, max_iter=1000)
        model.fit(X, y)
        
        # Vérifie la décroissance de la loss
        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]
        
        # Vérifie l'accuracy
        preds = model.predict(X)
        assert np.mean(preds == y) == 1.0  # Doit atteindre 100% de précision

    def test_real_data(self, real_data):
        """Teste sur données simulées complexes"""
        X, y = real_data
        model = LogisticRegression(max_iter=2000, learning_rate=0.01)
        model.fit(X, y)
        
        # Vérifie que le modèle apprend quelque chose d'utile
        train_accuracy = np.mean(model.predict(X) == y)
        assert train_accuracy > 0.8  # Doit faire mieux que random

    def test_non_linear_data(self, non_linear_data):
        """Teste sur données non-linéaires (limite du modèle linéaire)"""
        X, y = non_linear_data
        model = LogisticRegression(max_iter=3000, learning_rate=0.01)
        model.fit(X, y)
        
        # Le modèle linéaire ne peut pas bien apprendre XOR
        # Mais devrait faire mieux que 50% (hasard)
        assert 0.55 < np.mean(model.predict(X) == y) < 0.75

    def test_probability_output(self, binary_data):
        """Teste que les probabilités sont bien entre 0 et 1"""
        X, y = binary_data
        model = LogisticRegression()
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert np.all((probas >= 0) & (probas <= 1))
        assert probas.shape == y.shape

    def test_edge_cases(self, edge_case_data):
        """Teste différents cas limites"""
        # Cas avec une seule classe
        X, y = edge_case_data['single_class']
        model = LogisticRegression()
        model.fit(X, y)
        assert np.all(model.predict(X) == 0)
        
        # Données avec grandes valeurs
        X, y = edge_case_data['large_values']
        model.fit(X, y)
        assert model.predict(np.array([[1500]]))[0] in {0, 1}  # Doit pas crasher
        
        # Beaucoup de features
        X, y = edge_case_data['high_dim']
        model.fit(X, y)
        assert model.predict(X[0:1])[0] in {0, 1}

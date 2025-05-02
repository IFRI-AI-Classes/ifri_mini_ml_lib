import numpy as np
import pytest
from ifri_mini_lib.classification.decision_tree import DecisionTree

@pytest.fixture
def binary_data():
    """Données binaires simples"""
    X = np.array([[1, 2], [1.5, 3], [2, 1], [5, 6]])
    y = np.array([0, 0, 1, 1])
    return X, y

@pytest.fixture
def multi_data():
    """Données multiclasses complexes"""
    X = np.array([[1, 1], [1.1, 1.1], [2, 2], [2.1, 2.1], [3, 3], [3.1, 3.1]])
    y = np.array([0, 0, 1, 1, 2, 2])
    return X, y

@pytest.fixture
def noisy_data():
    """Données avec du bruit"""
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y

class TestDecisionTree:
    def test_initialization(self):
        """Teste différents paramètres d'initialisation"""
        for depth in [None, 1, 5]:
            model = DecisionTree(max_depth=depth)
            assert model.max_depth == depth

    def test_binary_classification(self, binary_data):
        """Teste la classification binaire"""
        X, y = binary_data
        model = DecisionTree(max_depth=2)
        model.fit(X, y)
        
        # Vérifie l'accuracy sur train
        preds = model.predict(X)
        assert np.mean(preds == y) == 1.0
        
        # Vérifie un point de test
        assert model.predict(np.array([[1.2, 2.3]]))[0] == 0

    def test_multi_class(self, multi_data):
        """Teste la classification multiclasse"""
        X, y = multi_data
        model = DecisionTree()
        model.fit(X, y)
        assert len(np.unique(model.predict(X))) == 3

    def test_noisy_data(self, noisy_data):
        """Teste sur données bruyantes"""
        X, y = noisy_data
        model = DecisionTree(max_depth=5)
        model.fit(X, y)
        assert np.mean(model.predict(X) == y) > 0.8

    def test_tree_structure(self, binary_data):
        """Teste la structure de l'arbre"""
        X, y = binary_data
        model = DecisionTree(max_depth=1)
        model.fit(X, y)
        
        assert isinstance(model.tree, dict)
        assert 'feature_index' in model.tree
        assert 'threshold' in model.tree
        assert isinstance(model.tree['left'], (dict, int))
        assert isinstance(model.tree['right'], (dict, int))

    def test_edge_cases(self):
        """Teste les cas limites"""
        # Une seule feature
        model = DecisionTree()
        model.fit(np.array([[1], [2], [3]]), np.array([0, 0, 1]))
        assert model.predict(np.array([[1.5]]))[0] == 0
        
        # Tous points identiques
        model.fit(np.array([[1], [1], [1]]), np.array([0, 0, 1]))
        assert model.predict(np.array([[1]]))[0] in {0, 1}
        
        # Données vides
        with pytest.raises(ValueError):
            model.fit(np.array([]), np.array([]))

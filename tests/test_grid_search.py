import pytest
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from ifri_mini_lib.model_selection import GridSearchCV

# Mock Model pour les tests
class MockClassifier(BaseEstimator):
    def __init__(self, param1=1, param2='a'):
        self.param1 = param1
        self.param2 = param2
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        return np.zeros(len(X))

# Données de test
X = np.random.rand(20, 2)
y = np.array([0]*10 + [1]*10)  # 10 samples per class

def test_grid_search_initialization():
    """Teste l'initialisation de GridSearchCV"""
    model = MockClassifier()
    param_grid = {'param1': [1, 2], 'param2': ['a', 'b']}
    grid = GridSearchCV(model, param_grid, accuracy_score)
    
    assert grid.k == 5
    assert not grid.stratified
    assert grid.best_score_ == -np.inf

def test_grid_search_fit():
    """Teste le fonctionnement de base de fit()"""
    model = MockClassifier()
    param_grid = {'param1': [1, 2], 'param2': ['a', 'b']}
    grid = GridSearchCV(model, param_grid, accuracy_score, k=2)
    
    grid.fit(X, y)
    
    assert grid.best_params_ is not None
    assert isinstance(grid.best_score_, float)
    assert 0 <= grid.best_score_ <= 1
    assert hasattr(grid.best_estimator_, 'fit')

def test_grid_search_with_stratified():
    """Teste le mode stratifié"""
    model = MockClassifier()
    param_grid = {'param1': [1]}
    grid = GridSearchCV(model, param_grid, accuracy_score, stratified=True, k=2)
    
    grid.fit(X, y)
    assert grid.best_score_ >= 0

def test_param_combinations(monkeypatch):
    """Test that all parameter combinations are evaluated"""
    # Setup
    model = MockClassifier()
    param_grid = {
        'param1': [1, 2, 3],  # 3 valeurs
        'param2': ['a', 'b']   # 2 valeurs
    }  # Total: 6 combinaisons
    grid = GridSearchCV(model, param_grid, accuracy_score, k=2)
    
    # Mock tracking
    call_counter = 0
    
    def mock_cv(model, X, y, metric, stratified, k):
        nonlocal call_counter
        call_counter += 1
        # Return dummy metrics
        return 0.75, 0.05  # mean, std
    
    # Apply mock
    monkeypatch.setattr(
        'ifri_mini_lib.model_selection.grid_searchCV.k_fold_cross_validation',
        mock_cv
    )
    
    # Execute
    grid.fit(X, y)
    
    # Verify
    assert call_counter == 6  # 3 x 2 combinations
    assert grid.best_score_ == 0.75  # From our mock
    assert len(grid.best_params_) == 2

def test_best_estimator():
    """Vérifie que le meilleur estimateur est bien entraîné"""
    model = MockClassifier()
    param_grid = {'param1': [1]}
    grid = GridSearchCV(model, param_grid, accuracy_score, k=2)
    
    grid.fit(X, y)
    assert hasattr(grid.best_estimator_, 'param1')
    assert grid.best_estimator_.param1 == 1
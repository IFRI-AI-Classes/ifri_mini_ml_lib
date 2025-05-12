import numpy as np
import pytest
from ifri_mini_ml_lib.model_selection.utils import BaseEstimatorMinimal
from ifri_mini_ml_lib.metrics.evaluation_metrics import accuracy
from ifri_mini_ml_lib.model_selection import RandomSearchCV  

# Mock classifier
class MockClassifier(BaseEstimatorMinimal):
    def __init__(self, param1=0, param2='x'):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))

# Données de test
X = np.random.rand(20, 3)
y = np.array([0]*10 + [1]*10)

def test_initialization():
    model = MockClassifier()
    param_grid = {'param1': [1, 2], 'param2': ['a', 'b']}
    search = RandomSearchCV(model, param_grid, scoring=accuracy, n_iter=3)
    
    assert search.k == 5
    assert search.best_score_ == -np.inf
    assert search.best_params_ is None

def test_fit_basic():
    model = MockClassifier()
    param_grid = {'param1': [1], 'param2': ['a']}
    search = RandomSearchCV(model, param_grid, scoring=accuracy, n_iter=1, k=2)
    
    search.fit(X, y)
    
    assert search.best_estimator_ is not None
    assert search.best_params_ == {'param1': 1, 'param2': 'a'}
    assert 0 <= search.best_score_ <= 1

def test_stratified_flag():
    model = MockClassifier()
    param_grid = {'param1': [1], 'param2': ['a']}
    search = RandomSearchCV(model, param_grid, scoring=accuracy, n_iter=1, k=2, stratified=True)
    
    search.fit(X, y)
    assert search.best_score_ >= 0

def test_random_sampling_respects_n_iter():
    model = MockClassifier()
    param_grid = {'param1': [1, 2, 3], 'param2': ['a', 'b']}  
    search = RandomSearchCV(model, param_grid, scoring=accuracy, n_iter=4, k=2)
    assert search.n_iter == 4

def test_random_search_with_mocked_cv(monkeypatch):
    model = MockClassifier()
    param_grid = {'param1': [1, 2], 'param2': ['a', 'b']} 
    n_iter = 3
    search = RandomSearchCV(model, param_grid, scoring=accuracy, n_iter=n_iter, k=2)

    call_counter = []

    def fake_cv(model, X, y, metric, stratified, k):
        call_counter.append(1)
        return 0.8, 0.1 

    monkeypatch.setattr(
        "ifri_mini_ml_lib.model_selection.random_searchCV.k_fold_cross_validation", 
        fake_cv
    )

    search.fit(X, y)
    assert len(call_counter) == n_iter
    assert search.best_score_ == 0.8
    assert isinstance(search.best_params_, dict)

class MockRegressor(BaseEstimatorMinimal):
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        return self

    def predict(self, X):
        # Prédit une constante (moyenne de y ou 0)
        return np.full(len(X), 0.5)
    
def mean_squared_error(y_true, y_pred):
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

def test_random_search_regression():
    # Générer des données linéaires
    np.random.seed(0)
    X = np.random.rand(100, 3)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

    # Définir un modèle de régression Ridge
    model = MockRegressor()

    # Grille d'hyperparamètres
    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'fit_intercept': [True, False]
    }

    search = RandomSearchCV(
        model=model,
        param_grid=param_grid,
        scoring=lambda y_true, y_pred: -mean_squared_error(y_true, y_pred), 
        n_iter=5,
        k=3,
        stratified=False,
        random_state=42
    )

    search.fit(X, y)

    # Vérifications
    assert search.best_estimator_ is not None
    assert isinstance(search.best_params_, dict)
    assert search.best_score_ < 0  # car on maximise le score négatif (MSE)

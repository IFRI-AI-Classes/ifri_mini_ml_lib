import numpy as np
import pytest
from ifri_mini_ml_lib.model_selection.utils import BaseEstimatorMinimal
from ifri_mini_ml_lib.metrics.evaluation_metrics import accuracy
from ifri_mini_ml_lib.model_selection import BaggingRegressor, BaggingClassifier 

class SimpleDecisionTreeRegressor(BaseEstimatorMinimal):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = np.mean(y)

    def predict(self, X):
        return np.full(len(X), self.tree_)
    
def mean_squared_error(y_true, y_pred):
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)

def test_bagging_regressor_training_and_prediction():
    # Générer des données de régression
    X = np.random.rand(200, 5)
    y = np.random.rand(200)

    # Instancier le modèle avec un arbre de décision
    base_model = SimpleDecisionTreeRegressor(max_depth=10)
    model = BaggingRegressor(base_model=base_model, n_estimators=10, random_state=0)

    # Entraîner
    model.fit(X, y)

    # Prédictions
    y_pred = model.predict(X)

    # Vérifications
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert not np.any(np.isnan(y_pred)), "Les prédictions contiennent des NaN"
    assert mean_squared_error(y, y_pred) < 500, "Erreur trop élevée"

def test_bagging_regressor_with_pretrained_models():
    # Générer des données
    X = np.random.rand(200, 5)
    y = np.random.rand(200)
    # Entraîner quelques modèles manuellement
    models = []
    for seed in [0, 1, 2]:
        model = SimpleDecisionTreeRegressor(max_depth=3)
        indices = np.random.choice(len(X), len(X), replace=True)
        model.fit(X[indices], y[indices])
        models.append(model)

    # Utiliser ces modèles dans BaggingRegressor
    bagger = BaggingRegressor(pretrained_models=models)
    y_pred = bagger.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (X.shape[0],)
    assert not np.any(np.isnan(y_pred))

def test_bagging_regressor_invalid_model():
    class InvalidModel:
        pass

    with pytest.raises(ValueError):
        model = BaggingRegressor(base_model=InvalidModel(), n_estimators=3)
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        model.fit(X, y)

class SimpleDecisionTreeClassifier(BaseEstimatorMinimal):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree_ = None

    def fit(self, X, y):
        self.tree_ = np.bincount(y).argmax()  
    def predict(self, X):
        return np.full(len(X), self.tree_)
    

def test_bagging_classifier_training_and_prediction():
    # Générer des données de classification
    X = np.random.rand(200, 5)
    y = np.random.randint(0, 2, 200)

    # Instancier le modèle
    base_model = SimpleDecisionTreeClassifier(max_depth=3)
    model = BaggingClassifier(base_model=base_model, n_estimators=5, random_state=0)

    # Entraîner
    model.fit(X, y)

    # Prédictions
    y_pred = model.predict(X)

    # Vérifications
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isin(y_pred, np.unique(y))), "Les classes prédites sont invalides"
    assert accuracy(y, y_pred) > 0.5

def test_bagging_classifier_with_pretrained_models():
    # Générer les données
    X = np.random.rand(200, 5)
    y = np.random.randint(0, 2, 200)

    # Entraîner manuellement des modèles
    models = []
    for seed in [0, 1, 2]:
        model = SimpleDecisionTreeClassifier(max_depth=2)
        indices = np.random.choice(len(X), len(X), replace=True)
        model.fit(X[indices], y[indices])
        models.append(model)

    # Créer BaggingClassifier avec ces modèles
    bagger = BaggingClassifier(pretrained_models=models)
    y_pred = bagger.predict(X)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(np.isin(y_pred, [0, 1]))

def test_bagging_classifier_invalid_model():
    class InvalidModel:
        pass

    with pytest.raises(ValueError):
        model = BaggingClassifier(base_model=InvalidModel(), n_estimators=3)
        X = np.random.rand(10, 2)
        y = np.random.randint(0, 2, size=10)
        model.fit(X, y)

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import make_regression, make_classification
from ifri_mini_lib.model_selection import BaggingRegressor, BaggingClassifier 

def test_bagging_regressor_training_and_prediction():
    # Générer des données de régression
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)

    # Instancier le modèle avec un arbre de décision
    base_model = DecisionTreeRegressor(max_depth=10, random_state=42)
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
    X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)

    # Entraîner quelques modèles manuellement
    models = []
    for seed in [0, 1, 2]:
        model = DecisionTreeRegressor(max_depth=3, random_state=seed)
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

def test_bagging_classifier_training_and_prediction():
    # Générer des données de classification
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)

    # Instancier le modèle
    base_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model = BaggingClassifier(base_model=base_model, n_estimators=5, random_state=0)

    # Entraîner
    model.fit(X, y)

    # Prédictions
    y_pred = model.predict(X)

    # Vérifications
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert np.all(np.isin(y_pred, np.unique(y))), "Les classes prédites sont invalides"
    assert accuracy_score(y, y_pred) > 0.8

def test_bagging_classifier_with_pretrained_models():
    # Générer les données
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    # Entraîner manuellement des modèles
    models = []
    for seed in [0, 1, 2]:
        model = DecisionTreeClassifier(max_depth=2, random_state=seed)
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

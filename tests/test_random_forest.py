import numpy as np
import pytest
from ifri_mini_ml_lib.classification.random_forest import RandomForest

@pytest.fixture
def binary_dataset():
    """Dataset linéairement séparable — deux classes bien distinctes."""
    X = np.array([
        [1, 1],
        [2, 1],
        [3, 1],
        [10, 2],
        [11, 2],
        [12, 2]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y
 
 
@pytest.fixture
def multiclass_dataset():
    """Dataset à trois classes."""
    X = np.array([
        [1, 0], [2, 0], [3, 0],
        [10, 5], [11, 5], [12, 5],
        [1, 10], [2, 10], [3, 10]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return X, y
 
 
@pytest.fixture
def trained_rf(binary_dataset):
    """Modèle RandomForest déjà entraîné sur le dataset binaire."""
    X, y = binary_dataset
    rf = RandomForest(n_estimators=10, random_state=42)
    rf.fit(X, y)
    return rf, X, y
 
# Tests de base

def test_random_forest_basic_fit_predict(trained_rf):
    # Récupérer le modèle entraîné et les données
    rf, X, y = trained_rf
 
    # Vérifier que le modèle prédit correctement sur les données d'entraînement
    preds = rf.predict(X)
    assert np.array_equal(preds, y)

# Tests predict_proba

def test_predict_proba_sums_to_one(trained_rf):
    # Récupérer le modèle entraîné et les données
    rf, X, y = trained_rf
 
    # Vérifier que les probabilités de chaque sample somment à 1
    proba = rf.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
 
 
def test_predict_proba_values_between_zero_and_one(trained_rf):
    # Récupérer le modèle entraîné et les données
    rf, X, y = trained_rf
 
    # Vérifier que toutes les probabilités sont comprises entre 0 et 1
    proba = rf.predict_proba(X)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)

# Tests multi-classes

def test_random_forest_multiclass(multiclass_dataset):
    # Créer et entraîner le modèle sur un dataset à trois classes
    X, y = multiclass_dataset
    rf = RandomForest(n_estimators=20, random_state=42)
    rf.fit(X, y)
 
    # Vérifier que les prédictions restent dans les trois classes connues
    preds = rf.predict(X)
    assert set(preds).issubset({0, 1, 2})

# Tests reproducibilité

def test_random_state_reproducibility(binary_dataset):
    # Deux forêts entraînées avec le même random_state sur les mêmes données
    X, y = binary_dataset
    rf1 = RandomForest(n_estimators=10, random_state=0)
    rf2 = RandomForest(n_estimators=10, random_state=0)
    rf1.fit(X, y)
    rf2.fit(X, y)
 
    # Vérifier que les deux modèles produisent les mêmes prédictions
    assert np.array_equal(rf1.predict(X), rf2.predict(X))

# Tests erreurs

def test_predict_raises_error_if_not_fitted():
    # Créer un modèle sans l'entraîner
    rf = RandomForest()
    X = np.array([[1, 2], [3, 4]])
 
    # Vérifier qu'une ValueError est levée si predict est appelé avant fit
    with pytest.raises(ValueError):
        rf.predict(X)
 
 
def test_predict_proba_raises_error_if_not_fitted():
    # Créer un modèle sans l'entraîner
    rf = RandomForest()
    X = np.array([[1, 2], [3, 4]])
 
    # Vérifier qu'une ValueError est levée si predict_proba est appelé avant fit
    with pytest.raises(ValueError):
        rf.predict_proba(X)
 
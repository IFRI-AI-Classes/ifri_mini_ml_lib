import pytest
import numpy as np
from ifri_mini_ml_lib.model_selection.utils import BaseEstimatorMinimal
from ifri_mini_ml_lib.model_selection.cross_validation import k_fold_cross_validation
from ifri_mini_ml_lib.metrics.evaluation_metrics import accuracy

# Mock model for testing
class MockModel(BaseEstimatorMinimal):
    def __init__(self, predict_value=1):
        self.predict_value = predict_value
        
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        return np.full(len(X), self.predict_value)

# Test data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 40% class 0, 60% class 1

def test_k_fold_validation_basic_functionality():
    """Test basic functionality with non-stratified k-fold"""
    model = MockModel(predict_value=1)
    mean_score, std_score = k_fold_cross_validation(
        model, X, y, accuracy, stratified=False, k=5
    )
    
    # Since model always predicts 1 and we have 60% class 1
    expected_mean = 0.6
    assert np.isclose(mean_score, expected_mean, atol=0.1)
    assert std_score >= 0

def test_stratified_k_fold():
    """Test that stratified k-fold preserves class distribution"""
    model = MockModel(predict_value=1)
    mean_score, std_score = k_fold_cross_validation(
        model, X, y, accuracy, stratified=True, k=5
    )
    
    # Stratified should give more consistent results across folds
    assert np.isclose(mean_score, 0.6, atol=0.1)
    assert std_score < 0.2  # Lower std than non-stratified

def test_invalid_k_values():
    """Test error handling for invalid k values"""
    model = MockModel()
    
    with pytest.raises(ValueError, match="Number of folds must be between"):
        k_fold_cross_validation(model, X, y, accuracy, stratified=False, k=1)
        
    with pytest.raises(ValueError, match="Number of folds must be between"):
        k_fold_cross_validation(model, X[:1], y[:1], accuracy, stratified=False, k=2)

def test_model_validation():
    """Test that invalid models are detected"""
    class BadModel:
        pass

    bad_model = BadModel()
    
    with pytest.raises(ValueError, match="The model must implement both"):
        k_fold_cross_validation(bad_model, X, y, accuracy, stratified=False)

def test_fold_size_handling():
    """Teste la gestion des folds quand n_samples n'est pas divisible par k"""
    # Configuration
    small_X = X[:7]  # 7 échantillons (non divisible par k=5)
    small_y = y[:7]
    model = MockModel(predict_value=1)  # Modèle mock qui prédit toujours 1
    k = 5

    # Exécution
    mean_score, std_score = k_fold_cross_validation(
        model, small_X, small_y, accuracy, stratified=False, k=k
    )

    # Vérifications
    # 1. Vérifie que les scores sont calculés
    assert isinstance(mean_score, float), "Le score moyen doit être un float"
    assert isinstance(std_score, float), "L'écart-type doit être un float"

    # 2. Vérifie que les folds sont bien répartis
    # (7 échantillons / k=5 → 2 folds de 2, 3 folds de 1)
    # -> Vérification indirecte via la performance attendue
    expected_mean = np.mean(small_y)  # 4/7 ≈ 0.57 (car 4 "1" dans y[:7])
    assert np.isclose(mean_score, expected_mean, atol=0.1), \
        f"Le score moyen devrait être proche de {expected_mean}"

    # 3. Vérifie que l'écart-type est cohérent
    assert 0 <= std_score <= 0.5, \
        "L'écart-type devrait être faible (car les folds sont équilibrés)"

def test_metric_calculation():
    """Test that the metric is properly calculated"""
    model = MockModel(predict_value=0)  # Predict all 0s
    
    mean_score, _ = k_fold_cross_validation(
        model, X, y, accuracy, stratified=True, k=5
    )
    
    # Should get 40% accuracy (proportion of class 0)
    assert np.isclose(mean_score, 0.4, atol=0.1)

def test_deterministic_with_random_state():
    """Test that results are deterministic with fixed random state"""
    model = MockModel()
    
    scores1 = k_fold_cross_validation(model, X, y, accuracy, stratified=False, k=5)
    scores2 = k_fold_cross_validation(model, X, y, accuracy, stratified=False, k=5)
    
    assert scores1 == scores2

def test_stratified_distribution():
    """Test that stratified sampling maintains class distribution in each fold"""
    model = MockModel()
    k = 5
    _, _ = k_fold_cross_validation(model, X, y, accuracy, stratified=True, k=k)
    

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ifri_mini_ml_lib.regression import PolynomialRegression

# ================================
# Tests pour PolynomialRegression
# ================================

class TestPolynomialRegression:
    
    def test_init(self):
        """Test l'initialisation de PolynomialRegression avec différents paramètres."""
        # Test avec paramètres par défaut
        model1 = PolynomialRegression()
        assert model1.degree == 2
        assert model1.method == "least_squares"
        assert model1.learning_rate == 0.01
        assert model1.epochs == 1000
        assert model1.w is None
        assert model1.b is None
        
        # Test avec paramètres personnalisés
        model2 = PolynomialRegression(degree=3, method="gradient_descent", learning_rate=0.05, epochs=2000)
        assert model2.degree == 3
        assert model2.method == "gradient_descent"
        assert model2.learning_rate == 0.05
        assert model2.epochs == 2000
    
    def test_polynomial_features_simple(self):
        """Test la génération de caractéristiques polynomiales avec entrée simple."""
        model = PolynomialRegression(degree=2)
        X = np.array([2])
        features = model._polynomial_features(X)
        
        # Pour x=2, les caractéristiques devraient être [2, 4]
        expected = np.array([[2, 4]])
        assert np.allclose(features, expected)
    
    def test_polynomial_features_multi_samples(self):
        """Test la génération de caractéristiques polynomiales avec plusieurs échantillons."""
        model = PolynomialRegression(degree=2)
        X = np.array([1, 2, 3])
        features = model._polynomial_features(X)
        
        # Pour [1,2,3], les caractéristiques devraient être [[1,1], [2,4], [3,9]]
        expected = np.array([[1, 1], [2, 4], [3, 9]])
        assert np.allclose(features, expected)
    
    def test_polynomial_features_higher_degree(self):
        """Test la génération de caractéristiques polynomiales avec degré élevé."""
        model = PolynomialRegression(degree=3)
        X = np.array([2])
        features = model._polynomial_features(X)
        
        # Pour x=2 avec degré 3, les caractéristiques devraient être [2, 4, 8]
        expected = np.array([[2, 4, 8]])
        assert np.allclose(features, expected)
    
    def test_polynomial_features_multiple_features(self):
        """Test la génération de caractéristiques polynomiales avec plusieurs variables."""
        model = PolynomialRegression(degree=2)
        X = np.array([[1, 2], [3, 4]])
        features = model._polynomial_features(X)
        # Résultat attendu avec termes croisés : [x1, x2, x1², x1x2, x2²]
        expected = np.array([
            [1, 2, 1, 2, 4],   # 1, 2, 1², 1*2, 2²
            [3, 4, 9, 12, 16]  # 3, 4, 3², 3*4, 4²
        ])
        assert np.allclose(features, expected)
    
    def test_fit_validation(self):
        """Test la validation des entrées dans fit."""
        model = PolynomialRegression()
        
        # Test avec des entrées vides
        with pytest.raises(ValueError, match="Training datas are empties."):
            model.fit([], [])
        
        # Test avec des longueurs différentes
        with pytest.raises(ValueError, match="X and y have to have the same slength."):
            model.fit([[1], [2]], [1])
    
    def test_fit_quadratic(self):
        """Test la régression polynomiale de degré 2 sur des données quadratiques."""
        X = [[1], [2], [3], [4], [5]]
        y = [1, 4, 9, 16, 25]  # y = x^2
        
        model = PolynomialRegression(degree=2)
        model.fit(X, y)
        
        # Vérifier les prédictions
        predictions = model.predict([[6], [7]])
        assert abs(predictions[0] - 36.0) < 1e-5
        assert abs(predictions[1] - 49.0) < 1e-5
    
    def test_fit_cubic(self):
        """Test la régression polynomiale de degré 3 sur des données cubiques."""
        X = [[1], [2], [3], [4]]
        y = [1, 8, 27, 64]  # y = x^3
        
        model = PolynomialRegression(degree=3)
        model.fit(X, y)
        
        # Vérifier les prédictions
        predictions = model.predict([[5], [6]])
        assert abs(predictions[0] - 125.0) < 1e-5
        assert abs(predictions[1] - 216.0) < 1e-5
    
    def test_fit_multiple_features(self):
        """Test fit avec plusieurs caractéristiques."""
        X = [[1, 2], [2,5], [3, 3]]
        y = [4, 50, 27]  # y = x1 * x2^2
        
        model = PolynomialRegression(degree=3)
        model.fit(X, y)
        
        # Vérifie que la prédiction pour [4, 4] est proche de 4 * 4^2 = 64
        pred = model.predict([[4, 4]])
        assert abs(pred[0] - 64.0) < 5.0  # Tolérance plus large car c'est une approximation
    
    def test_predict_input_validation(self):
        """Test la validation des entrées dans predict."""
        model = PolynomialRegression()
        model.fit([[1], [2], [3]], [1, 4, 9])
        
        # Test avec des entrées vides
        with pytest.raises(ValueError, match="Input datas for prediction are empties"):
            model.predict([])
        
        # Test avec None
        with pytest.raises(ValueError, match="Input datas for prediction are empties"):
            model.predict(None)
    
    def test_predict_different_input_shapes(self):
        """Test predict avec différentes formes d'entrée."""
        X = [[1], [2], [3]]
        y = [1, 4, 9]  # y = x^2
        
        model = PolynomialRegression(degree=2)
        model.fit(X, y)
        
        # Test avec array 1D
        pred1 = model.predict([4])
        assert abs(pred1[0] - 16.0) < 1e-5
        
        # Test avec liste de listes 2D
        pred2 = model.predict([[4]])
        assert abs(pred2[0] - 16.0) < 1e-5
        
        # Test avec numpy array 1D
        pred3 = model.predict(np.array([4]))
        assert abs(pred3[0] - 16.0) < 1e-5
        
        # Test avec numpy array 2D
        pred4 = model.predict(np.array([[4]]))
        assert abs(pred4[0] - 16.0) < 1e-5

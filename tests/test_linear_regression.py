import sys
import os
import pytest
import numpy as np

# Ajout explicite du chemin courant
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ifri_mini_ml_lib.regression import LinearRegression

# ================================
# Tests pour LinearRegression
# ================================

class TestLinearRegression:   
    def test_init(self):
        """Test l'initialisation de LinearRegression avec différents paramètres."""
        # Test avec paramètres par défaut
        model1 = LinearRegression()
        assert model1.method == "least_squares"
        assert model1.learning_rate == 0.01
        assert model1.epochs == 1000
        assert model1.w is None
        assert model1.b is None
        
        # Test avec paramètres personnalisés
        model2 = LinearRegression(method="gradient_descent", learning_rate=0.05, epochs=2000)
        assert model2.method == "gradient_descent"
        assert model2.learning_rate == 0.05
        assert model2.epochs == 2000
    
    def test_fit_input_validation(self):
        """Test la validation des entrées dans la méthode fit."""
        model = LinearRegression()
        
        # Test avec des entrées vides
        with pytest.raises(ValueError, match="X and y can't be empty."):
            model.fit([], [])
        
        # Test avec des longueurs différentes
        with pytest.raises(ValueError, match="X and y have to have the same lengths."):
            model.fit([[1], [2]], [1])
    
    def test_fit_reshape_y(self):
        """Test que fit peut gérer différentes formes de y."""
        X = [[1], [2], [3]]
        y1 = [2, 4, 6]  # Liste 1D
        y2 = [[2], [4], [6]]  # Liste 2D
        y3 = np.array([[2], [4], [6]])  # numpy array 2D
        
        model1 = LinearRegression()
        model2 = LinearRegression()
        model3 = LinearRegression()
        
        model1.fit(X, y1)
        model2.fit(X, y2)
        model3.fit(X, y3)
        
        # Vérifie que les trois modèles donnent le même résultat
        assert abs(model1.w - model2.w) < 1e-5
        assert abs(model1.b - model2.b) < 1e-5
        assert abs(model1.w - model3.w) < 1e-5
        assert abs(model1.b - model3.b) < 1e-5
    
    def test_fit_simple_linear_regression(self):
        """Test la régression linéaire simple avec des données parfaitement linéaires."""
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]  # y = 2x
        
        model = LinearRegression(method="least_squares")
        model.fit(X, y)
        
        # Vérifier que les coefficients sont corrects
        assert abs(model.w - 2.0) < 1e-5
        assert abs(model.b - 0.0) < 1e-5
        
        # Vérifier les prédictions
        predictions = model.predict([[6], [7]])
        assert abs(predictions[0] - 12.0) < 1e-5
        assert abs(predictions[1] - 14.0) < 1e-5
    
    def test_fit_multiple_linear_regression(self):
        """Test la régression linéaire multiple."""
        X = [[1, 2], [2, 1], [3, 3], [4, 2]]
        y = [5, 4, 9, 8]  # y = x1 + 2*x2
        
        model = LinearRegression(method="least_squares")
        model.fit(X, y)
        
        # Vérifier les prédictions
        predictions = model.predict([[5, 3], [2, 4]])
        assert abs(predictions[0] - 11.0) < 1e-5
        assert abs(predictions[1] - 10.0) < 1e-5
    
    def test_fit_with_gradient_descent(self):
        """Test la régression linéaire avec descente de gradient."""
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]  # y = 2x
        
        model = LinearRegression(method="gradient_descent", learning_rate=0.01, epochs=5000)
        model.fit(X, y)
        
        # Vérifier que les coefficients sont approximativement corrects
        assert abs(model.w - 2.0) < 0.1
        assert abs(model.b - 0.0) < 0.1
        
        # Test avec des données multivariées
        X_multi = [[1, 1], [2, 2], [3, 3]]
        y_multi = [3, 6, 9]  # y = x1 + 2*x2
        
        model_multi = LinearRegression(method="gradient_descent", learning_rate=0.01, epochs=5000)
        model_multi.fit(X_multi, y_multi)
        
        # Les prédictions doivent être approximativement correctes
        predictions = model_multi.predict([[4, 4]])
        assert abs(predictions[0] - 12.0) < 1.0
    
    def test_fit_invalid_method(self):
        """Test que fit lève une exception pour une méthode inconnue."""
        model = LinearRegression(method="invalid_method")
        with pytest.raises(ValueError, match="Méthode inconnue"):
            model.fit([[1], [2]], [1, 2])
    
    def test_fit_simple_zero_denominator(self):
        """Test _fit_simple avec un dénominateur nul."""
        # Création de données où tous les X sont identiques
        X = [[1], [1], [1]]
        y = [2, 3, 4]
        
        model = LinearRegression()
        with pytest.raises(ValueError, match="Division par zéro"):
            model.fit(X, y)
    
    def test_fit_simple_empty_data(self):
        """Test _fit_simple avec des données vides."""
        model = LinearRegression()
        with pytest.raises(ValueError, match="Input datas are empties"):
            # On passe directement à _fit_simple pour tester sa validation
            model._fit_simple(np.array([]), np.array([]))
    
    def test_predict_input_validation(self):
        """Test la validation des entrées dans predict."""
        model = LinearRegression()
        model.fit([[1], [2], [3]], [2, 4, 6])
        
        # Test avec des entrées vides
        with pytest.raises(ValueError, match="Input datas for prediction are empties"):
            model.predict([])
        
        # Test avec None
        with pytest.raises(ValueError, match="Input datas for prediction are empties"):
            model.predict(None)
    
    def test_predict_different_input_shapes(self):
        """Test predict avec différentes formes d'entrée."""
        X = [[1], [2], [3]]
        y = [2, 4, 6]  # y = 2x
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Test avec array 1D
        pred1 = model.predict([4])
        assert abs(pred1[0] - 8.0) < 1e-5
        
        # Test avec liste de listes 2D
        pred2 = model.predict([[4]])
        assert abs(pred2[0] - 8.0) < 1e-5
        
        # Test avec numpy array 1D
        pred3 = model.predict(np.array([4]))
        assert abs(pred3[0] - 8.0) < 1e-5
        
        # Test avec numpy array 2D
        pred4 = model.predict(np.array([[4]]))
        assert abs(pred4[0] - 8.0) < 1e-5
    
    def test_predict_multiple_features(self):
        """Test predict avec plusieurs caractéristiques."""
        X = [[1, 2], [2, 3], [3, 4]]
        y = [5, 8, 11]  # y = x1 + 2*x2
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédiction avec multiples caractéristiques
        pred = model.predict([[4, 5]])
        # La prédiction doit être proche de 4 + 2*5 = 14
        assert abs(pred[0] - 14.0) < 1.0

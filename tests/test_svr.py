import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ifri_mini_ml_lib.regression import SVR

# ================================
# Tests pour SVR
# ================================

class TestSVR:
    
    def test_init(self):
        """Test l'initialisation de SVR avec différents paramètres."""
        # Test avec paramètres par défaut
        model = SVR(C_reg=1.0, epsilon=0.1, kernel="lin")
        assert model._c_reg == 1.0
        assert model.eps == 0.1
        assert model._ker == "lin"
        assert model._c == 1
        assert model._deg == 3
        assert model._gamma == 1
        assert model._alpha == 0.01
        assert model._test_size == 0.2
        
        # Test avec paramètres personnalisés
        model2 = SVR(C_reg=2.0, epsilon=0.2, kernel="rbf", c=2, d=4, gamma=0.5, alpha=0.02, test_size=0.3)
        assert model2._c_reg == 2.0
        assert model2.eps == 0.2
        assert model2._ker == "rbf"
        assert model2._c == 2
        assert model2._deg == 4
        assert model2._gamma == 0.5
        assert model2._alpha == 0.02
        assert model2._test_size == 0.3
    
    def test_init_invalid_kernel(self):
        """Test l'initialisation avec un noyau invalide."""
        model = SVR(C_reg=1.0, epsilon=0.1, kernel="invalid_kernel")
        # Devrait par défaut utiliser "lin"
        assert model._ker == "lin"
    
    def test_str_repr(self):
        """Test les méthodes __str__ et __repr__."""
        model = SVR(C_reg=1.0, epsilon=0.1, kernel="lin")
        
        # Vérifie __str__
        str_output = str(model)
        assert "kernel:lin" in str_output
        assert "epsilon:0.1" in str_output
        assert "C_reg:1.0" in str_output
        
        # Vérifie __repr__
        repr_output = repr(model)
        assert "SVR(kernel:lin" in repr_output
        assert "epsilon:0.1" in repr_output
        assert "C_reg:1.0" in repr_output
    
    def test_kernel_getter_setter(self):
        """Test le getter et le setter pour kernel."""
        model = SVR(C_reg=1.0, epsilon=0.1, kernel="lin")
        
        # Vérifie le getter
        assert model.ker == "lin"
        
        # Teste le setter avec un noyau valide
        model.ker = "rbf"
        assert model.ker == "rbf"
        
        # Teste le setter avec un noyau invalide
        with pytest.raises(ValueError):
            model.ker = "invalid_kernel"
    
    def test_linear_kernel(self):
        """Test le noyau linéaire."""
        model = SVR(C_reg=1.0, epsilon=0.1, kernel="lin")
        
        # Test avec une seule matrice d'entrée
        X = np.array([[1, 2], [3, 4]])
        K = model.linear_kernel(X)
        expected = np.array([[5, 11], [11, 25]])  # [1,2]·[1,2]^T = 5, etc.
        assert np.allclose(K, expected)
        
        # Test avec deux matrices d'entrée
        Y = np.array([[5, 6], [7, 8]])
        K2 = model.linear_kernel(X, Y)
        expected2 = np.array([[17, 23], [39, 53]])  # [1,2]·[5,6]^T = 17, etc.
        assert np.allclose(K2, expected2)
    
    def test_polynomial_kernel(self):
        """Test le noyau polynomial."""
        model = SVR(C_reg=1.0, epsilon=0.1, kernel="poly", c=1, d=2)
        
        X = np.array([[1, 2], [3, 4]])
        
        # Test avec une seule matrice d'entrée
        K = model.polynomial_kernel(X)
        # ([1,2]·[1,2]^T + 1)^2 = (5+1)^2 = 36, etc.
        expected = np.array([[36, 144], [144, 676]])
        assert np.allclose(K, expected)
        
        # Test avec deux matrices d'entrée
        Y = np.array([[5, 6], [7, 8]])
        K2 = model.polynomial_kernel(X, Y)
        # ([1,2]·[5,6]^T + 1)^2 = (17+1)^2 = 324, etc.
        expected2 = np.array([[324, 576], [1600, 2916]])
        assert np.allclose(K2, expected2)
    
    def test_rbf_kernel(self):
        """Test le noyau RBF."""
        model = SVR(C_reg=1.0, epsilon=0.1, kernel="rbf", gamma=0.5)
        
        X = np.array([[1, 1], [2, 2]])
        
        # Test avec une seule matrice d'entrée
        K = model.rbf_kernel(X)
        # Pour les points [1,1] et [2,2], la distance carrée est (1-2)^2 + (1-2)^2 = 2
        # exp(-0.5 * 2) ≈ 0.3679
        expected = np.array([[1.0, np.exp(-0.5 * 2)], [np.exp(-0.5 * 2), 1.0]])
        assert np.allclose(K, expected)
        
        # Test avec deux matrices d'entrée
        Y = np.array([[3, 3], [4, 4]])
        K2 = model.rbf_kernel(X, Y)
        # Pour les points [1,1] et [3,3], la distance carrée est (1-3)^2 + (1-3)^2 = 8
        # exp(-0.5 * 8) ≈ 0.0183
        expected2 = np.array([
            [np.exp(-0.5 * 8), np.exp(-0.5 * 18)],
            [np.exp(-0.5 * 2), np.exp(-0.5 * 8)]
        ])
        assert np.allclose(K2, expected2)
    
    def test_sigmoid_kernel(self):
        """Test le noyau sigmoid."""
        model = SVR(C_reg=1.0, epsilon=0.1, kernel="sig", alpha=0.1, c=1)
        
        X = np.array([[1, 2], [3, 4]])
        
        # Test avec une seule matrice d'entrée
        K = model.sigmoid_kernel(X)
        # tanh(0.1 * ([1,2]·[1,2]^T) + 1) = tanh(0.1 * 5 + 1) ≈ tanh(1.5)
        expected = np.array([
            [np.tanh(0.1 * 5 + 1), np.tanh(0.1 * 11 + 1)],
            [np.tanh(0.1 * 11 + 1), np.tanh(0.1 * 25 + 1)]
        ])
        assert np.allclose(K, expected)
        
        # Test avec deux matrices d'entrée
        Y = np.array([[5, 6], [7, 8]])
        K2 = model.sigmoid_kernel(X, Y)
        expected2 = np.array([
            [np.tanh(0.1 * 17 + 1), np.tanh(0.1 * 23 + 1)],
            [np.tanh(0.1 * 39 + 1), np.tanh(0.1 * 53 + 1)]
        ])
        assert np.allclose(K2, expected2)
    
    def test_get_kernel(self):
        """Test la méthode get_kernel."""
        # Test pour chaque type de noyau
        kernels = ["lin", "poly", "rbf", "sig"]
        kernel_methods = [
            SVR.linear_kernel, 
            SVR.polynomial_kernel, 
            SVR.rbf_kernel, 
            SVR.sigmoid_kernel
        ]
        
        for ker, method in zip(kernels, kernel_methods):
            model = SVR(C_reg=1.0, epsilon=0.1, kernel=ker)
            kernel_func = model.get_kernel()
            # Vérifier que la fonction renvoyée est la bonne
            assert kernel_func.__name__ == method.__name__
    
    def test_fit_linear_kernel(self):
        """Test la méthode fit avec noyau linéaire."""
        # Données linéaires avec bruit
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])  # y ≈ 2x
        
        model = SVR(C_reg=10.0, epsilon=0.1, kernel="lin")
        model.fit(X, y)
        
        # Vérifier que les attributs nécessaires sont créés
        assert hasattr(model, 'alpha')
        assert hasattr(model, 'alpha_star')
        assert hasattr(model, 'b')
        assert hasattr(model, 'kernel_matrix')
        
        # Vérifier les prédictions
        predictions = model.predict(np.array([[6], [7]]))
        
        # Les prédictions devraient être proches de 12 et 14
        assert 11.0 < predictions[0] < 13.0
        assert 13.0 < predictions[1] < 15.0
    
    def test_fit_rbf_kernel(self):
        """Test la méthode fit avec noyau RBF."""
        # Données non linéaires
        X = np.array([[-2], [-1], [0], [1], [2]])
        y = np.array([4.1, 1.1, 0.2, 0.9, 4.2])  # y ≈ x^2
        
        model = SVR(C_reg=10.0, epsilon=0.1, kernel="rbf", gamma=1.0)
        model.fit(X, y)
        
        # Vérifier les prédictions
        predictions = model.predict(np.array([[-1.5], [1.5]]))
        
        # Les prédictions devraient être proches de 2.25 pour les deux points
        assert 1.5 < predictions[0] < 3.0
        assert 1.5 < predictions[1] < 3.0
    
    def test_predict_with_dataframe(self):
        """Test la méthode predict avec un DataFrame pandas."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # y = 2x
        
        model = SVR(C_reg=10.0, epsilon=0.1, kernel="lin")
        model.fit(X, y)
        
        # Créer un DataFrame pandas pour les prédictions
        X_test_df = pd.DataFrame({'x': [6, 7]})
        
        predictions = model.predict(X_test_df)
        
        # Les prédictions devraient être proches de 12 et 14
        assert 11.0 < predictions[0] < 13.0
        assert 13.0 < predictions[1] < 15.0
    
    def test_set_params_updated(self):
        """Test la méthode set_params avec un mapping amélioré."""
        model = SVR(C_reg=1.0, epsilon=0.1, kernel="lin")
        
        # Utilisation de c_reg (minuscule) devrait fonctionner
        model.set_params(c_reg=2.0, gamma=0.5)
        assert model._c_reg == 2.0
        assert model._gamma == 0.5
        
        # Utilisation de C_reg (majuscule) devrait fonctionner aussi avec le mapping
        model.set_params(C_reg=3.0, epsilon=0.2)
        assert model._c_reg == 3.0
        assert model.eps == 0.2  # eps mapping à epsilon

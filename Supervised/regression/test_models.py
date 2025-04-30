import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import sys
import warnings
import os

# Import des modules à tester
from SVR import SVR
from Regressions import LinearRegression, PolynomialRegression
from metrics import evaluate_model

class TestMetrics(unittest.TestCase):
    """Tests pour les fonctions de métriques."""
    
    def setUp(self):
        """Préparation des données pour les tests."""
        self.y_true = [10, 20, 30, 40, 50]
        self.y_pred = [12, 18, 32, 38, 52]
    
    def test_metrics_calculation(self):
        """Test du calcul correct des métriques."""
        metrics = evaluate_model(self.y_true, self.y_pred)
        
        # Vérification des métriques calculées manuellement
        errors = np.array(self.y_true) - np.array(self.y_pred)
        expected_mse = np.mean(errors ** 2)
        expected_rmse = np.sqrt(expected_mse)
        expected_mae = np.mean(np.abs(errors))
        
        self.assertAlmostEqual(metrics["MSE"], expected_mse, places=5)
        self.assertAlmostEqual(metrics["RMSE"], expected_rmse, places=5)
        self.assertAlmostEqual(metrics["MAE"], expected_mae, places=5)
    
    def test_r2_calculation(self):
        """Test spécifique pour le calcul du R²."""
        metrics = evaluate_model(self.y_true, self.y_pred)
        
        y_mean = np.mean(self.y_true)
        ss_total = np.sum((np.array(self.y_true) - y_mean) ** 2)
        ss_res = np.sum((np.array(self.y_true) - np.array(self.y_pred)) ** 2)
        expected_r2 = 1 - (ss_res / ss_total)
        
        self.assertAlmostEqual(metrics["R²"], expected_r2, places=5)
    
    def test_error_handling(self):
        """Test de la gestion des erreurs."""
        # Test avec des listes de longueurs différentes
        with self.assertRaises(ValueError):
            evaluate_model([1, 2, 3], [1, 2])
        
        # Test avec une division par zéro (tous les y_true identiques)
        metrics = evaluate_model([5, 5, 5], [4, 6, 5])
        self.assertEqual(metrics["R²"], 0)


class TestLinearRegression(unittest.TestCase):
    """Tests pour la classe LinearRegression."""
    
    def setUp(self):
        """Préparation des données pour les tests."""
        np.random.seed(42)
        self.X_simple = np.linspace(0, 10, 50)
        self.y_simple = 2 * self.X_simple + 3 + np.random.randn(50)
        
        # Données pour régression multiple
        self.X_multiple = np.array([[x, x**2] for x in self.X_simple])
        self.y_multiple = 2 * self.X_simple + 1.5 * self.X_simple**2 + 3 + np.random.randn(50)
    
    def test_initialization(self):
        """Test d'initialisation de la classe."""
        model = LinearRegression(method="least_squares")
        self.assertEqual(model.method, "least_squares")
        self.assertEqual(model.learning_rate, 0.01)
        self.assertEqual(model.epochs, 1000)
        self.assertEqual(model.w, [])
        self.assertEqual(model.b, 0)
    
    def test_simple_fit_least_squares(self):
        """Test de l'ajustement avec les moindres carrés pour une régression simple."""
        model = LinearRegression(method="least_squares")
        model.fit(self.X_simple.tolist(), self.y_simple.tolist())
        
        # Vérification que les paramètres ont été correctement calculés
        self.assertIsInstance(model.w, (int, float))
        self.assertIsInstance(model.b, (int, float))
        
        # Vérification que la prédiction fonctionne
        y_pred = model.predict(self.X_simple.tolist())
        self.assertEqual(len(y_pred), len(self.X_simple))
    
    def test_simple_fit_gradient_descent(self):
        """Test de l'ajustement avec la descente de gradient pour une régression simple."""
        model = LinearRegression(method="gradient_descent", learning_rate=0.001, epochs=100)
        model.fit(self.X_simple.tolist(), self.y_simple.tolist())
        
        # Vérification que les paramètres ont été correctement calculés
        self.assertIsInstance(model.w, (int, float))
        self.assertIsInstance(model.b, (int, float))
        
        # Vérification que la prédiction fonctionne
        y_pred = model.predict(self.X_simple.tolist())
        self.assertEqual(len(y_pred), len(self.X_simple))
    
    def test_multiple_fit_least_squares(self):
        """Test de l'ajustement avec les moindres carrés pour une régression multiple."""
        model = LinearRegression(method="least_squares")
        model.fit(self.X_multiple.tolist(), self.y_multiple.tolist())
        
        # Vérification que les paramètres ont été correctement calculés
        self.assertEqual(len(model.w), 2)  # Deux coefficients pour deux variables
        self.assertIsInstance(model.b, (int, float))
        
        # Vérification que la prédiction fonctionne
        y_pred = model.predict(self.X_multiple.tolist())
        self.assertEqual(len(y_pred), len(self.X_multiple))
    
    def test_error_handling(self):
        """Test de la gestion des erreurs."""
        model = LinearRegression()
        
        # Test avec des données vides
        with self.assertRaises(ValueError):
            model.fit([], [])
        
        # Test avec des données de tailles différentes
        with self.assertRaises(ValueError):
            model.fit([1, 2, 3], [1, 2])
        
        # Test de prédiction avec des données vides
        with self.assertRaises(ValueError):
            model.predict([])


class TestPolynomialRegression(unittest.TestCase):
    """Tests pour la classe PolynomialRegression."""
    
    def setUp(self):
        """Préparation des données pour les tests."""
        np.random.seed(42)
        self.X = np.linspace(-5, 5, 50)
        self.y = 2 * self.X**2 + 3 * self.X + 5 + np.random.randn(50) * 5
        
        # Conversion en format de liste pour les tests
        self.X_list = [[x] for x in self.X]
        self.y_list = self.y.tolist()
    
    def test_initialization(self):
        """Test d'initialisation de la classe."""
        model = PolynomialRegression(degree=3)
        self.assertEqual(model.degree, 3)
        self.assertEqual(model.method, "least_squares")
        self.assertEqual(model.w, [])
        self.assertEqual(model.b, 0)
    
    def test_polynomial_features(self):
        """Test de la génération des caractéristiques polynomiales."""
        model = PolynomialRegression(degree=2)
        X_poly = model._polynomial_features(self.X_list)
        
        # Vérification de la forme des caractéristiques
        self.assertEqual(len(X_poly), len(self.X_list))
        self.assertEqual(len(X_poly[0]), 2)  # x et x^2 pour un degré 2
        
        # Vérification des valeurs
        for i, x in enumerate(self.X_list):
            self.assertAlmostEqual(X_poly[i][0], x[0], places=5)
            self.assertAlmostEqual(X_poly[i][1], x[0]**2, places=5)
    
    def test_fit_and_predict(self):
        """Test de l'ajustement et de la prédiction."""
        model = PolynomialRegression(degree=2)
        
        # Tentative de fit avec des données appropriées
        try:
            model.fit(self.X_list, self.y_list)
            
            # Vérification de la prédiction
            y_pred = model.predict(self.X_list)
            self.assertEqual(len(y_pred), len(self.X_list))
            
            # Calcul des métriques
            metrics = evaluate_model(self.y_list, y_pred)
            # Vérification que R² est positif (modèle meilleur qu'une simple moyenne)
            self.assertGreater(metrics["R²"], 0)
            
        except Exception as e:
            self.fail(f"Le fit ou predict a échoué: {e}")


class TestSVR(unittest.TestCase):
    """Tests pour la classe SVR."""
    
    def setUp(self):
        """Préparation des données pour les tests."""
        np.random.seed(42)
        self.X = np.linspace(-5, 5, 50).reshape(-1, 1)
        self.y = np.sin(self.X.ravel()) + np.random.randn(50) * 0.2
    
    def test_initialization(self):
        """Test d'initialisation de la classe."""
        model = SVR(C_reg=10, epsilon=0.1, kernel="rbf")
        self.assertEqual(model._c_reg, 10)
        self.assertEqual(model.eps, 0.1)
        self.assertEqual(model._ker, "rbf")
    
    def test_kernel_functions(self):
        """Test des fonctions de noyau."""
        model = SVR(C_reg=1, epsilon=0.1, kernel="rbf")
        
        # Test du noyau linéaire
        K_lin = model.linear_kernel(self.X)
        self.assertEqual(K_lin.shape, (50, 50))
        
        # Test du noyau RBF
        K_rbf = model.rbf_kernel(self.X)
        self.assertEqual(K_rbf.shape, (50, 50))
        self.assertTrue(np.all(K_rbf <= 1.0))  # Les valeurs RBF sont <= 1
        self.assertTrue(np.all(K_rbf > 0))  # Les valeurs RBF sont > 0
    
    def test_kernel_setter(self):
        """Test du setter pour le kernel."""
        model = SVR(C_reg=1, epsilon=0.1, kernel="lin")
        
        # Test de changement vers un noyau valide
        model.ker = "rbf"
        self.assertEqual(model._ker, "rbf")
        
        # Test de changement vers un noyau invalide
        with self.assertRaises(ValueError):
            model.ker = "invalid_kernel"
    
    def test_fit_and_predict(self):
        """Test de l'ajustement et de la prédiction."""
        # Nous allons réduire la taille des données pour accélérer le test
        X_small = self.X[:20]
        y_small = self.y[:20]
        
        # Capture la sortie standard pendant l'exécution
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            model = SVR(C_reg=1, epsilon=0.1, kernel="lin")
            
            # Tentative de fit
            model.fit(X_small, y_small)
            
            # Vérification que des vecteurs de support ont été trouvés
            output = captured_output.getvalue()
            self.assertIn("Number of support vectors:", output)
            
            # Tentative de prédiction
            y_pred = model.predict(X_small)
            self.assertEqual(len(y_pred), len(X_small))
            
        except Exception as e:
            self.fail(f"Le fit ou predict a échoué: {e}")
        finally:
            sys.stdout = sys.__stdout__  # Restaure la sortie standard
    
    def test_preprocessing(self):
        """Test de prétraitement des données."""
        # Test avec un DataFrame pandas
        df_X = pd.DataFrame(self.X)
        model = SVR(C_reg=1, epsilon=0.1, kernel="lin")
        
        # Vérifier que le modèle peut prédire avec un DataFrame
        model.fit(self.X, self.y)
        y_pred_df = model.predict(df_X)
        y_pred_np = model.predict(self.X)
        
        # Les prédictions doivent être identiques
        np.testing.assert_array_almost_equal(y_pred_df, y_pred_np)


class TestSuiteIntégration(unittest.TestCase):
    """Tests d'intégration avec différents jeux de données."""
    
    def setUp(self):
        """Préparation des jeux de données pour les tests."""
        np.random.seed(42)
        
        # Jeu de données 1: Relation linéaire
        self.X1 = np.linspace(0, 10, 100).reshape(-1, 1)
        self.y1 = 2 * self.X1.ravel() + 3 + np.random.randn(100) * 2
        
        # Jeu de données 2: Relation polynomiale
        self.X2 = np.linspace(-5, 5, 100).reshape(-1, 1)
        self.y2 = 2 * self.X2.ravel()**2 + 3 * self.X2.ravel() + 5 + np.random.randn(100) * 5
        
        # Jeu de données 3: Relation non-linéaire
        self.X3 = np.linspace(-5, 5, 100).reshape(-1, 1)
        self.y3 = np.sin(self.X3.ravel()) + np.random.randn(100) * 0.2
    
    def test_performance_comparison(self):
        """Compare les performances des différents modèles sur différents jeux de données."""
        results = {}
        
        # Test sur le jeu de données linéaire
        print("\nTest sur données linéaires:")
        results["linear"] = self._evaluate_models_on_dataset(self.X1, self.y1)
        
        # Test sur le jeu de données polynomial
        print("\nTest sur données polynomiales:")
        results["polynomial"] = self._evaluate_models_on_dataset(self.X2, self.y2)
        
        # Test sur le jeu de données non-linéaire
        print("\nTest sur données non-linéaires:")
        results["nonlinear"] = self._evaluate_models_on_dataset(self.X3, self.y3)
        
        # Assertions sur les performances attendues
        # Linear regression devrait bien performer sur les données linéaires
        self.assertGreater(results["linear"]["LinearRegression"]["R²"], 0.8)
        
        # Polynomial regression devrait bien performer sur les données polynomiales
        self.assertGreater(results["polynomial"]["PolynomialRegression"]["R²"], 0.7)
        
        # SVR avec noyau RBF devrait bien performer sur les données non-linéaires
        self.assertGreater(results["nonlinear"]["SVR_rbf"]["R²"], 0.7)
    
    def _evaluate_models_on_dataset(self, X, y):
        """Évalue tous les modèles sur un jeu de données et retourne les métriques."""
        # Préparation des données au format liste pour les modèles qui le nécessitent
        X_list = X.tolist()
        y_list = y.tolist()
        
        results = {}
        
        # Test avec Linear Regression
        try:
            model_lr = LinearRegression(method="least_squares")
            model_lr.fit(X_list, y_list)
            y_pred_lr = model_lr.predict(X_list)
            metrics_lr = evaluate_model(y_list, y_pred_lr)
            results["LinearRegression"] = metrics_lr
            print(f"Linear Regression: R² = {metrics_lr['R²']:.4f}, RMSE = {metrics_lr['RMSE']:.4f}")
        except Exception as e:
            print(f"Erreur avec Linear Regression: {e}")
            results["LinearRegression"] = {"R²": 0, "RMSE": float('inf')}
        
        # Test avec Polynomial Regression
        try:
            model_pr = PolynomialRegression(degree=2, method="least_squares")
            model_pr.fit(X_list, y_list)
            y_pred_pr = model_pr.predict(self._convert_to_poly_input(X))
            metrics_pr = evaluate_model(y_list, y_pred_pr)
            results["PolynomialRegression"] = metrics_pr
            print(f"Polynomial Regression: R² = {metrics_pr['R²']:.4f}, RMSE = {metrics_pr['RMSE']:.4f}")
        except Exception as e:
            print(f"Erreur avec Polynomial Regression: {e}")
            results["PolynomialRegression"] = {"R²": 0, "RMSE": float('inf')}
        
        # Test avec SVR (noyau linéaire)
        try:
            model_svr_lin = SVR(C_reg=1, epsilon=0.1, kernel="lin")
            model_svr_lin.fit(X, y)
            y_pred_svr_lin = model_svr_lin.predict(X)
            metrics_svr_lin = evaluate_model(y, y_pred_svr_lin)
            results["SVR_lin"] = metrics_svr_lin
            print(f"SVR (linéaire): R² = {metrics_svr_lin['R²']:.4f}, RMSE = {metrics_svr_lin['RMSE']:.4f}")
        except Exception as e:
            print(f"Erreur avec SVR (linéaire): {e}")
            results["SVR_lin"] = {"R²": 0, "RMSE": float('inf')}
        
        # Test avec SVR (noyau RBF)
        try:
            model_svr_rbf = SVR(C_reg=1, epsilon=0.1, kernel="rbf")
            model_svr_rbf.fit(X, y)
            y_pred_svr_rbf = model_svr_rbf.predict(X)
            metrics_svr_rbf = evaluate_model(y, y_pred_svr_rbf)
            results["SVR_rbf"] = metrics_svr_rbf
            print(f"SVR (RBF): R² = {metrics_svr_rbf['R²']:.4f}, RMSE = {metrics_svr_rbf['RMSE']:.4f}")
        except Exception as e:
            print(f"Erreur avec SVR (RBF): {e}")
            results["SVR_rbf"] = {"R²": 0, "RMSE": float('inf')}
        
        return results
    
    def _convert_to_poly_input(self, X):
        """Convertit les entrées pour le modèle de régression polynomiale."""
        return [[x[0]] for x in X.tolist()]


class TestVisualization(unittest.TestCase):
    """Tests pour les visualisations."""
    
    def setUp(self):
        """Préparation des données pour les tests."""
        np.random.seed(42)
        self.X = np.linspace(-5, 5, 50).reshape(-1, 1)
        self.y = np.sin(self.X.ravel()) + np.random.randn(50) * 0.2
    
    def test_visualization_functions(self):
        """Test des fonctions de visualisation."""
        # Désactivation de l'affichage des graphiques pour les tests
        plt.ioff()
        
        # Test de visualisation pour SVR
        model_svr = SVR(C_reg=1, epsilon=0.1, kernel="rbf")
        model_svr.fit(self.X, self.y)
        y_pred_svr = model_svr.predict(self.X)
        
        # Création de graphique
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Tracé des données
        ax.scatter(self.X, self.y, color='blue', label='Données')
        ax.plot(self.X, y_pred_svr, color='red', label='Prédictions SVR')
        
        # Ajout d'un epsilon-tube
        sorted_idx = np.argsort(self.X.ravel())
        X_sorted = self.X[sorted_idx].ravel()
        y_pred_sorted = y_pred_svr[sorted_idx]
        ax.fill_between(X_sorted, y_pred_sorted - model_svr.eps, y_pred_sorted + model_svr.eps,
                         color='gray', alpha=0.3, label='epsilon-tube')
        
        ax.legend()
        ax.grid(True)
        
        # Sauvegarde du graphique pour vérification (facultatif)
        plt.savefig("test_svr_visualization.png")
        plt.close(fig)
        
        # Vérification que le fichier a été créé
        self.assertTrue(os.path.exists("test_svr_visualization.png"))
        
        # Nettoyage
        os.remove("test_svr_visualization.png")


def run_tests():
    """Exécute tous les tests."""
    # Désactivation des avertissements pour les tests
    warnings.filterwarnings("ignore")
    
    # Création de la suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajout des tests
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestLinearRegression))
    suite.addTests(loader.loadTestsFromTestCase(TestPolynomialRegression))
    suite.addTests(loader.loadTestsFromTestCase(TestSVR))
    suite.addTests(loader.loadTestsFromTestCase(TestSuiteIntégration))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualization))
    
    # Exécution des tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Affichage des résultats
    print(f"\nTests terminés avec {result.testsRun} tests exécutés.")
    print(f"Succès: {result.testsRun - len(result.errors) - len(result.failures)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
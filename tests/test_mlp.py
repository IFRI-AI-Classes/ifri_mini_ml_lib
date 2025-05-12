import pytest
import numpy as np
from ifri_mini_ml_lib.neural_networks import MLPClassifier as MLP  

class TestMLP:
    def setup_method(self):
        """Configuration commune pour les tests."""
        np.random.seed(42)
        # Création d'un petit jeu de données synthétique
        self.X = np.random.randn(100, 5)  # 100 échantillons, 5 features
        self.y = np.random.randint(0, 3, size=100)  # 3 classes
        
    def test_initialization(self):
        """Test de l'initialisation du MLP."""
        mlp = MLP(hidden_layer_sizes=(10, 5), random_state=42)
        assert mlp.hidden_layer_sizes == (10, 5)
        assert mlp.activation == "relu"
        assert mlp.solver == "sgd"
        assert mlp.trained is False
        
    def test_activation_functions(self):
        """Test des fonctions d'activation."""
        mlp = MLP()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Test ReLU
        assert np.allclose(mlp._relu(x), np.array([0.0, 0.0, 0.0, 1.0, 2.0]))
        
        # Test sigmoid
        sigmoid_expected = 1 / (1 + np.exp(-x))
        assert np.allclose(mlp._sigmoid(x), sigmoid_expected)
        
        # Test tanh
        assert np.allclose(mlp._tanh(x), np.tanh(x))
        
        # Test leaky_relu
        leaky_expected = np.where(x > 0, x, 0.01 * x)
        assert np.allclose(mlp._leaky_relu(x), leaky_expected)
        
    def test_activation_derivatives(self):
        """Test des dérivées des fonctions d'activation."""
        mlp = MLP()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Test ReLU derivative
        assert np.allclose(mlp._relu_derivative(x), np.array([0.0, 0.0, 0.0, 1.0, 1.0]))
        
        # Test sigmoid derivative
        sigmoid_x = mlp._sigmoid(x)
        sigmoid_der_expected = sigmoid_x * (1 - sigmoid_x)
        assert np.allclose(mlp._sigmoid_derivative(x), sigmoid_der_expected)
        
        # Test tanh derivative
        tanh_der_expected = 1 - np.power(np.tanh(x), 2)
        assert np.allclose(mlp._tanh_derivative(x), tanh_der_expected)
        
        # Test leaky_relu derivative
        leaky_der_expected = np.where(x > 0, 1.0, 0.01)
        assert np.allclose(mlp._leaky_relu_derivative(x), leaky_der_expected)
        
    def test_forward_pass(self):
        """Test de la propagation avant."""
        mlp = MLP(hidden_layer_sizes=(3,), random_state=42)
        mlp._initialize_weights(n_features=5, n_outputs=3)
        
        # On vérifie juste que la forme des sorties est correcte
        activations, layer_inputs = mlp._forward_pass(self.X)
        
        assert len(activations) == 3  # Entrée + 1 couche cachée + sortie
        assert len(layer_inputs) == 2  # 1 couche cachée + sortie
        
        assert activations[0].shape == (100, 5)  # Entrée
        assert activations[1].shape == (100, 3)  # Couche cachée
        assert activations[2].shape == (100, 3)  # Sortie
        
        # Vérification que les activations de sortie sont des probabilités
        assert np.allclose(np.sum(activations[2], axis=1), np.ones(100))
        
    def test_fit_predict(self):
        """Test de l'entraînement et de la prédiction."""
        # Utilisons un jeu de données très simple pour tester l'apprentissage
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])  # XOR
        
        # Utilisons un réseau plus grand pour ce problème non linéaire
        mlp = MLP(
            hidden_layer_sizes=(10, 10),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42
        )
        
        mlp.fit(X, y)
        
        # Vérifier que le modèle est marqué comme entraîné
        assert mlp.trained is True
        
        # Vérifier que l'historique des pertes est enregistré
        assert len(mlp.loss_history) > 0
        
        # Prédiction
        y_pred = mlp.predict(X)
        
        # Pour XOR, on ne s'attend pas à une prédiction parfaite avec si peu d'itérations
        # mais on veut au moins vérifier que la forme est correcte
        assert y_pred.shape == y.shape
        
    def test_different_optimizers(self):
        """Test avec différents optimiseurs."""
        optimizers = ["sgd", "momentum", "rmsprop", "adam"]
        
        for optimizer in optimizers:
            mlp = MLP(
                hidden_layer_sizes=(5,),
                solver=optimizer,
                max_iter=10,  # Peu d'itérations pour le test
                random_state=42
            )
            
            mlp.fit(self.X, self.y)
            y_pred = mlp.predict(self.X)
            
            assert y_pred.shape == self.y.shape
            
    def test_invalid_parameters(self):
        """Test avec des paramètres invalides."""
        # Test avec une fonction d'activation invalide
        with pytest.raises(ValueError):
            mlp = MLP(activation="invalid_activation")
            mlp.fit(self.X, self.y)
            
        # Test avec un optimiseur invalide
        with pytest.raises(ValueError):
            mlp = MLP(solver="invalid_solver")
            mlp.fit(self.X, self.y)
            
    def test_predict_without_training(self):
        """Test de prédiction sans entraînement préalable."""
        mlp = MLP()
        
        with pytest.raises(ValueError):
            mlp.predict(self.X)
            
        with pytest.raises(ValueError):
            mlp.predict_proba(self.X)
            
        with pytest.raises(ValueError):
            mlp.score(self.X, self.y)
            
    def test_early_stopping(self):
        """Test de l'arrêt précoce."""
        mlp = MLP(
            hidden_layer_sizes=(5,),
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            max_iter=50,
            random_state=42
        )
        
        mlp.fit(self.X, self.y)
        
        # Vérifier que l'historique des pertes de validation est enregistré
        assert len(mlp.val_loss_history) > 0
        assert len(mlp.val_loss_history) <= 50  # Ne doit pas dépasser max_iter
        
    def test_different_activations(self):
        """Test avec différentes fonctions d'activation."""
        activations = ["sigmoid", "relu", "tanh", "leaky_relu"]
        
        for activation in activations:
            mlp = MLP(
                hidden_layer_sizes=(5,),
                activation=activation,
                max_iter=10,  # Peu d'itérations pour le test
                random_state=42
            )
            
            mlp.fit(self.X, self.y)
            y_pred = mlp.predict(self.X)
            
            assert y_pred.shape == self.y.shape

    def test_predict_proba(self):
        """Test de la prédiction de probabilités."""
        mlp = MLP(hidden_layer_sizes=(5,), max_iter=10, random_state=42)
        mlp.fit(self.X, self.y)
        
        probas = mlp.predict_proba(self.X)
        
        # Vérifier que les probabilités somment à 1
        assert np.allclose(np.sum(probas, axis=1), np.ones(len(self.X)))
        assert probas.shape == (len(self.X), 3)  # 3 classes
        
        # Vérifier que les prédictions correspondent aux probabilités max
        y_pred = mlp.predict(self.X)
        y_pred_from_proba = np.argmax(probas, axis=1)
        assert np.all(y_pred == y_pred_from_proba)
        
    def test_score(self):
        """Test de la fonction score."""
        mlp = MLP(hidden_layer_sizes=(5,), max_iter=10, random_state=42)
        mlp.fit(self.X, self.y)
        
        accuracy = mlp.score(self.X, self.y)
        
        # Vérifier que l'accuracy est entre 0 et 1
        assert 0 <= accuracy <= 1
        
        # Calculer manuellement l'accuracy et vérifier
        y_pred = mlp.predict(self.X)
        manual_accuracy = np.mean(y_pred == self.y)
        assert accuracy == manual_accuracy

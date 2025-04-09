import numpy as np
from typing import List, Tuple, Callable, Optional, Union, Dict


class MLP:
    """
    Multi-Layer Perceptron avec différentes fonctions d'activation et optimiseurs
    """
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = "relu",
        solver: str = "sgd",
        alpha: float = 0.0001,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        momentum: float = 0.9
    ):
        """
        Initialise un réseau MLP
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple
            Les tailles des couches cachées
        activation : str
            Fonction d'activation ('sigmoid', 'relu', 'tanh')
        solver : str
            L'algorithme d'optimisation ('sgd', 'adam', 'rmsprop', 'momentum')
        alpha : float
            Paramètre de régularisation L2
        batch_size : int
            Taille des batchs pour l'entraînement
        learning_rate : float
            Taux d'apprentissage
        max_iter : int
            Nombre maximum d'itérations
        shuffle : bool
            Si True, mélange les données à chaque epoch
        random_state : int or None
            Graine pour la reproductibilité
        beta1 : float
            Paramètre pour Adam (décroissance exponentielle du premier moment)
        beta2 : float
            Paramètre pour Adam (décroissance exponentielle du second moment)
        epsilon : float
            Valeur pour éviter la division par zéro
        momentum : float
            Paramètre pour l'optimiseur momentum
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = momentum
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sélection des fonctions d'activation
        self.activation_functions = {
            'sigmoid': self._sigmoid,
            'relu': self._relu,
            'tanh': self._tanh,
        }
        
        self.activation_derivatives = {
            'sigmoid': self._sigmoid_derivative,
            'relu': self._relu_derivative,
            'tanh': self._tanh_derivative,
        }
        
        # Sélection de la fonction d'activation et de sa dérivée
        if activation not in self.activation_functions:
            raise ValueError(f"Activation '{activation}' non reconnue. Utilisez 'sigmoid', 'relu', ou 'tanh'.")
        
        self.activation_func = self.activation_functions[activation]
        self.activation_derivative = self.activation_derivatives[activation]
        
        # Initialisation des poids
        self.weights = []
        self.biases = []
        self.n_layers = None
        self.n_outputs = None
        
        # Pour les optimiseurs
        self.velocity_weights = []  # Pour Momentum
        self.velocity_biases = []
        self.m_weights = []  # Pour Adam
        self.m_biases = []
        self.v_weights = []  # Pour Adam
        self.v_biases = []
        self.t = 1  # Timestep pour Adam
        
        self.loss_history = []
        self.trained = False
    
    def _initialize_weights(self, n_features: int, n_outputs: int) -> None:
        """
        Initialise les poids et biais du réseau
        
        Parameters:
        -----------
        n_features : int
            Nombre de features en entrée
        n_outputs : int
            Nombre de classes en sortie
        """
        # Dimensions des couches
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]
        self.n_layers = len(layer_sizes) - 1
        self.n_outputs = n_outputs
        
        # Initialisation des poids avec Xavier/Glorot
        for i in range(self.n_layers):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros(layer_sizes[i + 1]))
            
            # Initialisation pour les optimiseurs
            self.velocity_weights.append(np.zeros_like(self.weights[-1]))  # Pour Momentum
            self.velocity_biases.append(np.zeros_like(self.biases[-1]))
            self.m_weights.append(np.zeros_like(self.weights[-1]))  # Pour Adam
            self.m_biases.append(np.zeros_like(self.biases[-1]))
            self.v_weights.append(np.zeros_like(self.weights[-1]))  # Pour Adam
            self.v_biases.append(np.zeros_like(self.biases[-1]))
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Fonction d'activation softmax
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Fonction d'activation sigmoïde
        """
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Dérivée de la fonction sigmoïde
        """
        return x * (1 - x)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """
        Fonction d'activation ReLU
        """
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Dérivée de la fonction ReLU
        """
        return np.where(x > 0, 1, 0)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """
        Fonction d'activation tanh
        """
        return np.tanh(x)
    
    def _tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Dérivée de la fonction tanh
        """
        return 1 - np.power(x, 2)
    
    def _forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Propagation avant pour calculer les activations
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Données d'entrée
            
        Returns:
        --------
        activations : Liste des activations pour chaque couche
        layer_inputs : Liste des entrées pour chaque couche (avant activation)
        """
        activations = [X]
        layer_inputs = []
        
        # Passe à travers toutes les couches sauf la dernière
        for i in range(self.n_layers - 1):
            layer_input = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            layer_inputs.append(layer_input)
            activation = self.activation_func(layer_input)
            activations.append(activation)
        
        # Couche de sortie avec softmax pour la classification
        last_layer_input = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        layer_inputs.append(last_layer_input)
        
        # Utiliser softmax pour la classification
        output_activation = self._softmax(last_layer_input)
        activations.append(output_activation)
        
        return activations, layer_inputs
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule l'entropie croisée avec régularisation L2
        
        Parameters:
        -----------
        y_true : np.ndarray, shape (n_samples, n_classes)
            Les labels en one-hot encoding
        y_pred : np.ndarray, shape (n_samples, n_classes)
            Les prédictions du modèle
            
        Returns:
        --------
        loss : float
            La valeur de la perte
        """
        m = y_true.shape[0]
        # Entropie croisée
        log_likelihood = -np.sum(y_true * np.log(np.clip(y_pred, 1e-10, 1.0))) / m
        
        # Régularisation L2
        l2_reg = 0
        for w in self.weights:
            l2_reg += np.sum(np.square(w))
        l2_reg *= self.alpha / (2 * m)
        
        return log_likelihood + l2_reg
    
    def _backward_pass(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        activations: List[np.ndarray], 
        layer_inputs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Rétropropagation du gradient
        
        Parameters:
        -----------
        X : np.ndarray
            Données d'entrée
        y : np.ndarray
            Labels en one-hot encoding
        activations : Liste des activations pour chaque couche
        layer_inputs : Liste des entrées pour chaque couche
            
        Returns:
        --------
        gradients_w : Liste des gradients pour les poids
        gradients_b : Liste des gradients pour les biais
        """
        m = X.shape[0]
        gradients_w = [None] * self.n_layers
        gradients_b = [None] * self.n_layers
        
        # Gradient de la couche de sortie
        delta = activations[-1] - y  # Dérivée de l'entropie croisée avec softmax
        
        # Rétropropagation du gradient à travers les couches
        for i in range(self.n_layers - 1, -1, -1):
            # Calcul du gradient pour les poids et biais de la couche i
            gradients_w[i] = np.dot(activations[i].T, delta) / m + self.alpha * self.weights[i]
            gradients_b[i] = np.mean(delta, axis=0)
            
            # Rétropropagation du delta (sauf pour la première couche)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(activations[i])
        
        return gradients_w, gradients_b
    
    def _update_weights_sgd(self, gradients_w: List[np.ndarray], gradients_b: List[np.ndarray]) -> None:
        """
        Met à jour les poids avec la descente de gradient stochastique
        """
        for i in range(self.n_layers):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def _update_weights_momentum(self, gradients_w: List[np.ndarray], gradients_b: List[np.ndarray]) -> None:
        """
        Met à jour les poids avec la descente de gradient avec momentum
        """
        for i in range(self.n_layers):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] - self.learning_rate * gradients_w[i]
            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] - self.learning_rate * gradients_b[i]
            
            self.weights[i] += self.velocity_weights[i]
            self.biases[i] += self.velocity_biases[i]
    
    def _update_weights_rmsprop(self, gradients_w: List[np.ndarray], gradients_b: List[np.ndarray]) -> None:
        """
        Met à jour les poids avec RMSProp
        """
        decay_rate = 0.9
        
        for i in range(self.n_layers):
            # Mise à jour des accumulateurs
            self.v_weights[i] = decay_rate * self.v_weights[i] + (1 - decay_rate) * np.square(gradients_w[i])
            self.v_biases[i] = decay_rate * self.v_biases[i] + (1 - decay_rate) * np.square(gradients_b[i])
            
            # Mise à jour des poids
            self.weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.v_weights[i]) + self.epsilon)
            self.biases[i] -= self.learning_rate * gradients_b[i] / (np.sqrt(self.v_biases[i]) + self.epsilon)
    
    def _update_weights_adam(self, gradients_w: List[np.ndarray], gradients_b: List[np.ndarray]) -> None:
        """
        Met à jour les poids avec Adam
        """
        for i in range(self.n_layers):
            # Mise à jour des moments
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * gradients_w[i]
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * gradients_b[i]
            
            # Mise à jour des moments du second ordre
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * np.square(gradients_w[i])
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * np.square(gradients_b[i])
            
            # Correction du biais
            m_weights_corrected = self.m_weights[i] / (1 - self.beta1 ** self.t)
            m_biases_corrected = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_weights_corrected = self.v_weights[i] / (1 - self.beta2 ** self.t)
            v_biases_corrected = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            # Mise à jour des poids
            self.weights[i] -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon)
        
        self.t += 1
    
    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """
        Convertit les labels en représentation one-hot
        
        Parameters:
        -----------
        y : np.ndarray de shape (n_samples,)
            Les labels à encoder
            
        Returns:
        --------
        one_hot : np.ndarray de shape (n_samples, n_classes)
            Les labels encodés
        """
        n_samples = len(y)
        one_hot = np.zeros((n_samples, self.n_outputs))
        one_hot[np.arange(n_samples), y.astype(int)] = 1
        return one_hot
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP':
        """
        Entraîne le MLP sur les données fournies
        
        Parameters:
        -----------
        X : np.ndarray de shape (n_samples, n_features)
            Les données d'entraînement
        y : np.ndarray de shape (n_samples,)
            Les labels cibles
            
        Returns:
        --------
        self : objet
            Le MLP entraîné
        """
        # Conversion des arrays
        X = np.array(X, dtype=float)
        y = np.array(y)
        
        # Déterminer le nombre de classes
        n_samples, n_features = X.shape
        n_outputs = len(np.unique(y))
        
        # Initialisation des poids
        self._initialize_weights(n_features, n_outputs)
        
        # One-hot encoding des labels
        y_one_hot = self._one_hot_encode(y)
        
        # Méthode de mise à jour selon l'optimiseur choisi
        update_methods = {
            'sgd': self._update_weights_sgd,
            'momentum': self._update_weights_momentum,
            'rmsprop': self._update_weights_rmsprop,
            'adam': self._update_weights_adam
        }
        
        if self.solver not in update_methods:
            raise ValueError(f"Optimiseur '{self.solver}' non reconnu.")
        
        update_weights = update_methods[self.solver]
        
        # Entraînement sur plusieurs époques
        self.loss_history = []
        
        for _ in range(self.max_iter):
            # Mélange des données si demandé
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y_one_hot[indices]
            else:
                X_shuffled = X
                y_shuffled = y_one_hot
            
            # Entraînement par mini-batchs
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Propagation avant
                activations, layer_inputs = self._forward_pass(X_batch)
                
                # Calcul de la perte
                loss = self._compute_loss(y_batch, activations[-1])
                self.loss_history.append(loss)
                
                # Rétropropagation
                gradients_w, gradients_b = self._backward_pass(X_batch, y_batch, activations, layer_inputs)
                
                # Mise à jour des poids
                update_weights(gradients_w, gradients_b)
        
        self.trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les classes pour les échantillons X
        
        Parameters:
        -----------
        X : np.ndarray de shape (n_samples, n_features)
            Les données pour lesquelles on veut faire des prédictions
            
        Returns:
        --------
        y_pred : np.ndarray de shape (n_samples,)
            Les classes prédites
        """
        if not self.trained:
            raise ValueError("Le modèle doit être entraîné avant de pouvoir faire des prédictions.")
        
        X = np.array(X, dtype=float)
        activations, _ = self._forward_pass(X)
        y_pred = np.argmax(activations[-1], axis=1)
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités pour chaque classe
        
        Parameters:
        -----------
        X : np.ndarray de shape (n_samples, n_features)
            Les données pour lesquelles on veut faire des prédictions
            
        Returns:
        --------
        probas : np.ndarray de shape (n_samples, n_classes)
            Les probabilités pour chaque classe
        """
        if not self.trained:
            raise ValueError("Le modèle doit être entraîné avant de pouvoir faire des prédictions.")
        
        X = np.array(X, dtype=float)
        activations, _ = self._forward_pass(X)
        return activations[-1]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Retourne la précision du modèle sur les données fournies
        
        Parameters:
        -----------
        X : np.ndarray de shape (n_samples, n_features)
            Les données de test
        y : np.ndarray de shape (n_samples,)
            Les vrais labels
            
        Returns:
        --------
        accuracy : float
            La précision du modèle
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)



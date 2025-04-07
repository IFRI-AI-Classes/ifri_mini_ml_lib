import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        """
        Initialise le modèle de régression logistique.
        
        Paramètres:
        - learning_rate: Taux d'apprentissage pour la descente de gradient
        - max_iter: Nombre maximum d'itérations
        - tol: Tolérance pour le critère d'arrêt
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _sigmoid(self, z):
        """Fonction sigmoïde qui transforme les valeurs en probabilités entre 0 et 1."""
        return 1 / (1 + np.exp(-z))
    
    def _initialize_parameters(self, n_features):
        """Initialise les paramètres du modèle (poids et biais)."""
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def _compute_loss(self, y_true, y_pred):
        """Calcule la fonction de perte (log loss)."""
        # Petite valeur epsilon pour éviter le log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        """
        Entraîne le modèle sur les données en utilisant la descente de gradient.
        
        Paramètres:
        - X: Matrice de caractéristiques (n_samples, n_features)
        - y: Vecteur cible (n_samples,)
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        
        for i in range(self.max_iter):
            # Calcul des prédictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # Calcul des gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calcul et stockage de la perte
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Critère d'arrêt si la perte ne diminue plus suffisamment
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                break
    
    def predict_proba(self, X):
        """Retourne les probabilités prédites pour chaque classe."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """Retourne les prédictions de classe (0 ou 1) basées sur un seuil."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
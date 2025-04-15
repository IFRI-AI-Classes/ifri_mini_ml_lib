import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, 
                 early_exaggeration=12.0, learning_rate=200.0, 
                 n_iter=1000, min_grad_norm=1e-7, 
                 random_state=None, verbose=0):
        """
        Implémentation from scratch de t-SNE.
        
        Paramètres:
        -----------
        n_components : int, (défaut: 2)
            Dimension de l'espace embarqué
            
        perplexity : float, (défaut: 30)
            Contrôle le nombre de voisins locaux considérés
            
        early_exaggeration : float, (défaut: 12.0)
            Facteur d'exagération initial pour séparer les clusters
            
        learning_rate : float, (défaut: 200.0)
            Taux d'apprentissage pour la descente de gradient
            
        n_iter : int, (défaut: 1000)
            Nombre maximal d'itérations
            
        min_grad_norm : float, (défaut: 1e-7)
            Seuil minimal de la norme du gradient pour continuer
            
        random_state : int ou None, (défaut: None)
            Graine pour le générateur aléatoire
            
        verbose : int, (défaut: 0)
            Niveau de verbosité (0: silencieux, 1: progress, 2: détaillé)
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.min_grad_norm = min_grad_norm
        self.random_state = random_state
        self.verbose = verbose
        
        # Résultats
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _euclidean_distance(self, X):
        """Calcule la matrice des distances euclidiennes carrées entre tous les points."""
        sum_X = np.sum(np.square(X), axis=1)
        distances = np.add(-2 * np.dot(X, X.T), sum_X).T + sum_X
        np.fill_diagonal(distances, 0.0)
        return distances
    
    def _binary_search_perplexity(self, distances, perplexity, tol=1e-5, max_iter=50):
        """Trouve les sigma appropriés pour obtenir la perplexité souhaitée."""
        n_samples = distances.shape[0]
        P = np.zeros((n_samples, n_samples))
        beta = np.ones((n_samples, 1))
        log_perplexity = np.log(perplexity)
        
        # On ignore la diagonale (distance à soi-même = 0)
        for i in range(n_samples):
            beta_min = -np.inf
            beta_max = np.inf
            dist_i = distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))]
            
            for _ in range(max_iter):
                # Calcul des probabilités conditionnelles
                P_i = np.exp(-dist_i * beta[i])
                sum_Pi = np.sum(P_i)
                
                if sum_Pi == 0:
                    sum_Pi = 1e-8
                
                # Calcul de l'entropie
                H = np.log(sum_Pi) + beta[i] * np.sum(dist_i * P_i) / sum_Pi
                P_i = P_i / sum_Pi
                
                # Ajustement de beta (précision binaire)
                H_diff = H - log_perplexity
                if np.abs(H_diff) < tol:
                    break
                
                if H_diff > 0:
                    beta_min = beta[i].copy()
                    if beta_max == np.inf:
                        beta[i] *= 2.0
                    else:
                        beta[i] = (beta[i] + beta_max) / 2.0
                else:
                    beta_max = beta[i].copy()
                    if beta_min == -np.inf:
                        beta[i] /= 2.0
                    else:
                        beta[i] = (beta[i] + beta_min) / 2.0
            
            # Remplir la matrice P
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))] = P_i
        
        return P
    
    def _compute_joint_probabilities(self, X, perplexity):
        """Calcule les probabilités jointes p_ij."""
        # Calcul des distances euclidiennes carrées
        distances = self._euclidean_distance(X)
        
        # Calcul des probabilités conditionnelles
        P = self._binary_search_perplexity(distances, perplexity)
        
        # Symétrisation et normalisation
        P = (P + P.T) / (2.0 * P.shape[0])
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_low_dimensional_probabilities(self, Y):
        """Calcule les probabilités q_ij en basse dimension."""
        distances = self._euclidean_distance(Y)
        inv_distances = 1.0 / (1.0 + distances)
        np.fill_diagonal(inv_distances, 0.0)
        Q = inv_distances / np.sum(inv_distances)
        Q = np.maximum(Q, 1e-12)
        return Q
    
    def _compute_gradient(self, P, Q, Y):
        """Calcule le gradient de la divergence KL par rapport à l'embedding."""
        n = Y.shape[0]
        gradient = np.zeros_like(Y)
        
        # Calcul des termes (p_ij - q_ij) * (1 + ||y_i - y_j||²)^-1
        dist = 1.0 / (1.0 + self._euclidean_distance(Y))
        pq_diff = (P - Q) * dist
        
        # Calcul du gradient
        for i in range(n):
            gradient[i] = 4.0 * np.sum((Y[i] - Y) * pq_diff[:, i][:, np.newaxis], axis=0)
        
        return gradient
    
    def _compute_kl_divergence(self, P, Q):
        """Calcule la divergence KL entre P et Q."""
        return np.sum(P * np.log(P / Q))
    
    def fit(self, X):
        """Fit le modèle aux données X."""
        n_samples = X.shape[0]
        
        # Vérification des données
        if n_samples < 3 * self.perplexity:
            raise ValueError(f"Le nombre d'échantillons ({n_samples}) doit être au moins 3 * perplexity ({3*self.perplexity})")
        
        if self.verbose:
            print("Calcul des probabilités jointes P...")
        
        # Calcul des P en haute dimension
        P = self._compute_joint_probabilities(X, self.perplexity)
        P *= self.early_exaggeration
        
        # Initialisation aléatoire de Y
        Y = 1e-4 * np.random.randn(n_samples, self.n_components).astype(np.float32)
        
        # Variables pour l'optimisation
        previous_gradient = np.zeros_like(Y)
        gains = np.ones_like(Y)
        
        if self.verbose:
            print("Optimisation de l'embedding...")
        
        # Optimisation
        for i in range(self.n_iter):
            # Calcul des Q en basse dimension
            Q = self._compute_low_dimensional_probabilities(Y)
            
            # Calcul du gradient
            gradient = self._compute_gradient(P, Q, Y)
            grad_norm = np.linalg.norm(gradient)
            
            # Mise à jour avec momentum
            gains = (gains + 0.2) * ((gradient > 0) != (previous_gradient > 0)) + \
                    (gains * 0.8) * ((gradient > 0) == (previous_gradient > 0))
            gains = np.clip(gains, 0.01, np.inf)
            
            previous_gradient = gradient.copy()
            Y -= self.learning_rate * (gains * gradient)
            
            # Centrage des données
            Y = Y - np.mean(Y, axis=0)
            
            # Calcul de la divergence KL
            kl_div = self._compute_kl_divergence(P, Q)
            
            # Réduction de l'exagération après 100 itérations
            if i == 100:
                P /= self.early_exaggeration
            
            # Affichage des informations
            if self.verbose >= 1 and i % 100 == 0:
                print(f"Iteration {i}: KL divergence = {kl_div:.4f}, Gradient norm = {grad_norm:.4f}")
                
                if grad_norm < self.min_grad_norm:
                    if self.verbose:
                        print(f"Arrêt prématuré à l'itération {i}: norme du gradient trop faible")
                    break
        
        # Sauvegarde des résultats
        self.embedding_ = Y
        self.kl_divergence_ = kl_div
        self.n_iter_ = i + 1
        
        return self
    
    def fit_transform(self, X):
        """Fit le modèle aux données et retourne l'embedding."""
        self.fit(X)
        return self.embedding_


    def generate_test_data(n_samples=300, case='blobs', random_state=None):
        """Génère des données de test."""
        if random_state:
            np.random.seed(random_state)
        
        if case == 'blobs':
            # Données groupées en clusters
            centers = np.array([[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]])
            X = np.vstack([center + np.random.randn(n_samples//4, 3)*0.3 for center in centers])
            y = np.repeat(np.arange(4), n_samples//4)
        elif case == 'swiss_roll':
            # Données en forme de rouleau suisse
            t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
            X = np.vstack([t * np.cos(t), 10 * np.random.rand(n_samples), t * np.sin(t)]).T
            y = (t // np.pi).astype(int)
        else:
            # Données linéairement séparables
            X = np.random.randn(n_samples, 3)
            X[:n_samples//2] += 1
            X[n_samples//2:] -= 1
            y = np.zeros(n_samples)
            y[n_samples//2:] = 1
        
        # Normalisation
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return X, y

    def plot_results(X, y, title, ax=None):
        """Visualise les résultats en 2D ou 3D."""
        if ax is None:
            fig = plt.figure()
            if X.shape[1] == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        
        if X.shape[1] == 2:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.7)
        ax.set_title(title)
        ax.grid(True)


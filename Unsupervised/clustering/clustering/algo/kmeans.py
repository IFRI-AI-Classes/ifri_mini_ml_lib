import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt 
# sys.path.append('../utils')
from utils.utils import euclidean_distance

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='random', random_state=None):
        """Implémentation de l'algorithme k-means."""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X):
        """Initialise les centroïdes."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.init == 'random':
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[indices]
        elif self.init == 'k-means++':
            self.centroids = [X[np.random.randint(X.shape[0])]]
            for _ in range(1, self.n_clusters):
                distances = np.array([min([euclidean_distance(c,x) for c in self.centroids]) for x in X])
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                random_float = np.random.rand()
                for i, prob in enumerate(cumulative_probabilities):
                    if random_float < prob:
                        self.centroids.append(X[i])
                        break
            self.centroids = np.array(self.centroids)
        else:
            raise ValueError("init must be 'random' or 'k-means++'")

    def fit(self, X):
        """Entraîne le modèle k-means sur les données X."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._initialize_centroids(X)

        for _ in range(self.max_iter):
            distances = np.array([[euclidean_distance(x, centroid) for centroid in self.centroids] for x in X])
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

    def predict(self, X):
        """Prédit les clusters pour de nouvelles données."""
        distances = np.array([[euclidean_distance(x, centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """Applique la méthode fit puis la méthode predict."""
        self.fit(X)
        return self.predict(X)
    
    def plot_clusters(self, X):
        """Visualise les clusters dans un espace 2D."""
        if X.shape[1] != 2:
            print("Avertissement : La visualisation n'est possible qu'en 2D.")
            return

        plt.figure(figsize=(8, 6))
        for i in range(self.n_clusters):
            plt.scatter(X[self.labels == i, 0], X[self.labels == i, 1], label=f'Cluster {i}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', s=200, color='black', label='Centroids')
        plt.title('Clusters KMeans')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

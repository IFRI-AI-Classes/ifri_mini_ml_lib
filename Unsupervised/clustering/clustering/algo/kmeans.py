import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt 
from utils.utils import euclidean_distance

class KMeans:
    """
    KMeans Class: Custom implementation of the K-Means unsupervised clustering algorithm.

    Definition:
    ------------
    K-Means is an unsupervised learning algorithm that groups data into k clusters by minimizing 
    intra-cluster distances. It iteratively assigns points to the nearest centroid and updates 
    the centroids based on the current assignments.

    Constructor Arguments:
    -----------------------
    - n_clusters (int): Number of clusters (k) to form.
    - max_iter (int): Maximum number of iterations for the algorithm.
    - tol (float): Tolerance to declare convergence (based on centroid shifts).
    - init (str): Initialization method for centroids ('random' or 'k-means++').
    - random_state (int): Seed value for reproducibility.

    Main Methods:
    --------------
    - _initialize_centroids(X): Initializes centroids based on the chosen method.
    - fit(X): Trains the KMeans model on dataset X.
    - predict(X): Predicts the cluster index for each sample in X.
    - fit_predict(X): Combines fit() and predict(), returns cluster labels.
    - plot_clusters(X): Displays the clusters in 2D space (only if X has 2 features).
    """
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='random', random_state=None):
        """
        Initializes the KMeans clustering model with specified parameters.

        Parameters:
        -------------
        - n_clusters (int): Number of clusters to form. Default is 3.
        - max_iter (int): Maximum number of iterations allowed for convergence. Default is 300.
        - tol (float): Tolerance value used to check convergence based on centroid movement. Default is 1e-4.
        - init (str): Method to initialize centroids. Either 'random' or 'k-means++'. Default is 'random'.
        - random_state (int or None): Seed value for random number generator to ensure reproducibility. Default is None.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X):
    """
    Initializes the centroids for KMeans based on the chosen initialization method.

    Parameters:
    -------------
    - X (ndarray): Input data of shape (n_samples, n_features).

    Behavior:
    ----------
    - If init == 'random': Randomly selects k samples from X as initial centroids.
    - If init == 'k-means++': Implements the k-means++ strategy to spread out initial centroids.
    
    Raises:
    -------
    - ValueError: If the init method is not 'random' or 'k-means++'.
    """
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
        """
        Fits the KMeans model to the data X by iteratively updating cluster assignments and centroids.

        Parameters:
        -------------
        - X (ndarray): Input data of shape (n_samples, n_features).

        Procedure:
        -----------
        - Assigns each point to the nearest centroid.
        - Updates centroids as the mean of assigned points.
        - Repeats until centroids converge or max_iter is reached.
        """
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
        """Predicts clusters for new data."""
        distances = np.array([[euclidean_distance(x, centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """Apply the fit method then the predict method."""
        self.fit(X)
        return self.predict(X)
    
    def plot_clusters(self, X):
        """Visualizes clusters in 2D space."""
        if X.shape[1] != 2:
            print("Warning: Visualization is only possible in 2D.")
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

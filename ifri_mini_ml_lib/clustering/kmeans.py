import numpy as np
from .utils import euclidean_distance
from matplotlib import pyplot as plt

class KMeans:
    """
    KMeans Class: Custom implementation of the K-Means unsupervised clustering algorithm.
    
    Description:
    ------------
    Custom implementation of the K-Means unsupervised clustering algorithm.  
    It partitions data into k clusters by minimizing intra-cluster distances, iteratively assigning points to the nearest centroid and updating centroids accordingly.

    Arguments:
        n_clusters (int): Number of clusters to form.
        max_iter (int): Maximum iterations for convergence.
        tol (float): Threshold to declare convergence based on centroid movement.
        init (str): Centroid initialization method ('random' or 'k-means++').
        random_state (int or None): Seed for reproducibility.

    Example:
    >>> from ifri_mini_ml_lib.clustering import KMeans
    >>> kmeans = KMeans(n_clusters=3, max_iter=300, tol=1e-4, init='k-means++', random_state=42)
    >>> kmeans.fit(X)
    >>> labels = kmeans.predict(X)
    >>> kmeans.plot_clusters(X)
    """
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, init='random', random_state=None):
        """
        Description:
        ------------
        Initializes the KMeans clustering model with the specified parameters.

        Arguments:
        -----------
        - n_clusters (int): Number of clusters to form. Default is 3.
        - max_iter (int): Maximum number of iterations allowed. Default is 300.
        - tol (float): Convergence tolerance based on centroid movement. Default is 1e-4.
        - init (str): Method to initialize centroids ('random' or 'k-means++'). Default is 'random'.
        - random_state (int or None): Seed for random number generator. Default is None.

        Functions:
        -----------
        - Stores parameters and initializes centroids and labels attributes.

        Example:
        ---------
        kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
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
        Description:
        ------------
        Initializes centroids for KMeans clustering using the specified initialization method.

        Arguments:
        -----------
        - X (ndarray): Input data array of shape (n_samples, n_features).

        Functions:
        -----------
        - If 'random', selects k random samples as centroids.
        - If 'k-means++', selects centroids to spread out initial points.
        - Raises ValueError if init method is invalid.

        Example:
        ---------
        self._initialize_centroids(X)
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
        Description:
        ------------
        Fits the KMeans model to data X by iteratively assigning points to clusters and updating centroids until convergence or max iterations.

        Arguments:
        -----------
        - X (ndarray): Input data array of shape (n_samples, n_features).

        Functions:
        -----------
        - Initializes centroids.
        - Assigns each point to nearest centroid.
        - Updates centroids as mean of assigned points.
        - Stops when centroid shifts are below tolerance or max_iter reached.

        Example:
        ---------
        kmeans.fit(X)
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
        """
        Description:
        ------------
        Predicts the nearest cluster index for each sample in X based on trained centroids.

        Arguments:
        -----------
        - X (ndarray): New data samples to predict, shape (n_samples, n_features).

        Functions:
        -----------
        - Computes distances to centroids.
        - Returns index of closest centroid for each sample.

        Example:
        ---------
        labels = kmeans.predict(X_new)
        """
        distances = np.array([[euclidean_distance(x, centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """
        Description:
        ------------
        Convenience method that fits the model to X and returns the cluster labels.

        Arguments:
        -----------
        - X (ndarray): Input data array.

        Functions:
        -----------
        - Calls fit(X) then predict(X).

        Example:
        ---------
        labels = kmeans.fit_predict(X)
        """
        self.fit(X)
        return self.predict(X)

    def plot_clusters(self, X):
        """
        Description:
        ------------
        Visualizes the clusters for KMeans, supporting 1D, 2D, and 3D data.
        For higher dimensions, PCA is applied to reduce to 3D.

        Arguments:
        -----------
        - X (ndarray): Data array.

        Functions:
        -----------
        - Plots points colored by cluster assignment.
        - Marks centroids with a distinct symbol ('x').
        - Uses PCA for dimensionality reduction if > 3D.

        Exemple: 
        - kmeans.plot_clusters(data)
        """
        n_features = X.shape[1]

        if n_features > 3:
            from ifri_mini_ml_lib.preprocessing.dimensionality_reduction.pca import PCA
            pca = PCA(n_component=3)
            X = pca.fit_transform(X)
            n_features = 3
            print("Warning: Data reduced to 3D using PCA for visualization.")

        fig = plt.figure(figsize=(8, 6))

        # case 1
        if n_features == 1:
            ax = fig.add_subplot(111)
            for i in range(self.n_clusters):
                cluster_data = X[self.labels == i]
                ax.scatter(cluster_data[:, 0], np.zeros_like(cluster_data[:, 0]), label=f'Cluster {i}')
            ax.scatter(self.centroids[:, 0], np.zeros_like(self.centroids[:, 0]), marker='x', s=200, color='black', label='Centroids')
            ax.set_yticks([])
            ax.set_xlabel('Feature 1')
            ax.set_title('KMeans Clusters (1D)')

        # case 2D
        elif n_features == 2:
            ax = fig.add_subplot(111)
            for i in range(self.n_clusters):
                cluster_data = X[self.labels == i]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')
            ax.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', s=200, color='black', label='Centroids')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title('KMeans Clusters (2D)')
            
        # Case 3D
        elif n_features == 3:
            ax = fig.add_subplot(111, projection='3d')
            for i in range(self.n_clusters):
                cluster_data = X[self.labels == i]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], label=f'Cluster {i}')
            ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], marker='x', s=200, color='black', label='Centroids')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            ax.set_title('KMeans Clusters (3D)')

        plt.legend()
        plt.show()

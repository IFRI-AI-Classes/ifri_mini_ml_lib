import numpy as np
from .utils import euclidean_distance  # Import function euclidean_distance
from matplotlib import pyplot as plt

class DBSCAN:
    """
    Description:
    ------------
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that identifies clusters based on the density of points. It groups nearby points (defined by epsilon and min_samples) and labels outliers as noise.

    Arguments:
        eps (float): The maximum radius to consider two points as neighbors.
        min_samples (int): The minimum number of points to form a cluster.

    Example:
    >>> from ifri_mini_ml_lib.clustering import DBSCAN
    >>> dbscan = DBSCAN(eps=0.5, min_samples=5)
    >>> labels = dbscan.fit_predict(data)
    >>> dbscan.plot_clusters(data)
    """

    def __init__(self, eps=0.5, min_samples=5):
        """
        Initializes the DBSCAN parameters.

        Arguments:
            eps (float): The maximum radius to consider two points as neighbors.
            min_samples (int): The minimum number of points to form a cluster.

        Example:
        >>> dbscan = DBSCAN(eps=0.5, min_samples=5)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None  # Cluster labels

    def fit_predict(self, data):
        """
        Performs DBSCAN clustering on the provided data.

        Arguments:
            data (numpy.ndarray): The data to cluster (n_samples, n_features).

        Example:
        >>> labels = dbscan.fit_predict(data)
        """
        self.labels = np.full(len(data), -1)  # Initialize all points as noise
        cluster_id = 0

        for i in range(len(data)):
            if self.labels[i] != -1:
                continue  # Point already visited

            # Find the neighbors of the current point
            neighbors = self._region_query(data, i)

            if len(neighbors) < self.min_samples:
                # Not a central point, remains noise
                continue

            # New cluster
            self._expand_cluster(data, i, cluster_id, neighbors)
            cluster_id += 1

        return self.labels

    def _region_query(self, data, point_index):
        """
        Finds the neighbors of a point within a given radius.

        Arguments:
            data (numpy.ndarray): The data.
            point_index (int): The point index.

        Example:
        >>> neighbors = self._region_query(data, 5)
        """
        neighbors = []
        for i in range(len(data)):
            if euclidean_distance(data[point_index], data[i]) <= self.eps:
                neighbors.append(i)
        return neighbors
        
    def _expand_cluster(self, data, point_index, cluster_id, neighbors):
        """
        Extends a cluster from a core point.

        Arguments:
            data (numpy.ndarray): The data.
            point_index (int): The index of the core point.
            cluster_id (int): The ID of the current cluster.
            neighbors (list): The indices of the core point's neighbors.

        Example:
        >>> self._expand_cluster(data, 10, 0, neighbors)
        """
        self.labels[point_index] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_index = neighbors[i]
            if self.labels[neighbor_index] == -1:
                # Go to neighor
                self.labels[neighbor_index] = cluster_id

                # Find neighors to neighor
                new_neighbors = self._region_query(data, neighbor_index)
                if len(new_neighbors) >= self.min_samples:
                    # Adds new neighbors to the list of neighbors to visit
                    neighbors += set(new_neighbors)
            i += 1

    def plot_clusters(self, data):
        """
        Plots the resulting clusters after calling fit_predict().
        Supports 1D, 2D and 3D data natively. For data with more than 3 features, applies PCA reduction to 3D.

        Arguments:
            data (numpy.ndarray): Input data (n_samples, n_features >= 1).

        Example:
        >>> dbscan.plot_clusters(data)
        """
        max_dimensions = 3
        n_features = data.shape[1]

        # PCA reduction if > 3D
        if n_features > max_dimensions:
            from ifri_mini_ml_lib.preprocessing.dimensionality_reduction.pca import PCA
            reducer = PCA(n_component=3)
            data = reducer.fit_transform(data)
            print(f"Warning: PCA reduction applied from {n_features}D to {max_dimensions}D")
            n_features = max_dimensions  # Update after PCA

        unique_labels = set(self.labels)
        colors = plt.colormaps.get_cmap('tab10')
        fig = plt.figure(figsize=(10, 7))

        # Case 1D
        if n_features == 1:
            ax = fig.add_subplot(111)
            for label in unique_labels:
                if label == -1:
                    color = 'k'
                    label_name = 'Noise'
                else:
                    color = colors(label)
                    label_name = f'Cluster {label}'
                cluster_points = data[self.labels == label]
                ax.scatter(cluster_points[:, 0], np.zeros_like(cluster_points[:, 0]), c=[color], label=label_name)
            ax.set_yticks([])
            ax.set_xlabel("Feature 1")
            ax.set_title("DBSCAN Clustering Result (1D)" )

        # Case 2D
        elif n_features == 2:
            ax = fig.add_subplot(111)
            for label in unique_labels:
                if label == -1:
                    color = 'k'
                    label_name = 'Noise'
                else:
                    color = colors(label)
                    label_name = f'Cluster {label}'
                cluster_points = data[self.labels == label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=label_name)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_title("DBSCAN Clustering Result (2D)" if n_features <= max_dimensions else f"DBSCAN Clustering Result (PCA reduced from {n_features}D to 2D)")

        # Case 3D
        elif n_features == 3:
            ax = fig.add_subplot(111, projection='3d')
            for label in unique_labels:
                if label == -1:
                    color = 'k'
                    label_name = 'Noise'
                else:
                    color = colors(label)
                    label_name = f'Cluster {label}'
                cluster_points = data[self.labels == label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=[color], label=label_name, s=30)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
            ax.set_title("DBSCAN Clustering Result (3D)" if n_features <= max_dimensions else f"DBSCAN Clustering Result (PCA reduced from {n_features}D to 3D)")

        plt.legend()
        plt.grid(True)
        plt.show()

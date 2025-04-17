import numpy as np
from utils.utils import euclidean_distance  # Import function euclidean_distance

class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) Class.

    DBSCAN is a clustering algorithm that identifies clusters based on the density of points in a given space. It works by grouping together nearby points (defined by an epsilon radius and a minimum number of points) and identifying points labeled as "noise" that do not belong to any cluster.

    The algorithm is particularly effective for arbitrary-shaped clusters and is robust to noise and outliers.

    Attributes:
   - epsilon (float): The radius of the neighborhood around a point.
   - min_samples (int): The minimum number of points required for a group to be considered a cluster.
   - labels_ (array): Assigns a label to each data point, with -1 representing noise points.

   Methods:
   - fit(X): Applies the DBSCAN algorithm on a dataset X.
   - _expand_cluster(X, labels, point_idx, cluster_id): Expands a cluster starting from a given point.
   """

    def __init__(self, eps=0.5, min_samples=5):
        """
        Initializes the DBSCAN parameters.

        Args:
        eps (float): The maximum radius to consider two points as neighbors.
        min_samples (int): The minimum number of points to form a cluster.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None  # Cluster labels

    def fit_predict(self, data):
        """
        Performs DBSCAN clustering on the provided data.

        Args:
        data (numpy.ndarray): The data to cluster (n_samples, n_features).

        Returns:
        numpy.ndarray: The cluster labels for each point (-1 for noise).
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

        Args:
        data (numpy.ndarray): The data.
        point_index (int): The point index.

        Returns:
        list: The neighbor indices.
        """
        neighbors = []
        for i in range(len(data)):
            if euclidean_distance(data[point_index], data[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, data, point_index, cluster_id, neighbors):
        """
        Extends a cluster from a center point.

        Args:
        data (numpy.ndarray): The data.
        point_index (int): The index of the center point.
        cluster_id (int): The ID of the current cluster.
        neighbors (list): The indices of the center point's neighbors.
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
                    neighbors += new_neighbors
            i += 1

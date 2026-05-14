import numpy as np
import pandas as pd


class LOF:

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.k = n_neighbors            # Number of neighbors to consider
        self.contamination = contamination  # Expected proportion of outliers
        self.X_train = None

    
    def fit(self, X):
        """Stores the training data for neighbor-based computations."""
        self.X_train = np.array(X)
        return self

    def _euclidean(self, a, b):
        """Computes the Euclidean distance between two points a and b."""
        return np.sqrt(np.sum((a - b) ** 2))

    def _k_neighbors(self, x, X):
        """Finds the k nearest neighbors of x in X.

        Returns:
            knn_idx: indices of the k nearest neighbors
            k_dist: distance to the k-th nearest neighbor
            distances: all distances from x to every point in X
        """
        # Calculate all the distances between x and the points in X
       
        distances = [self._euclidean(x, xi) for xi in X]
        distances = np.array(distances)

        sorted_idx = np.argsort(distances)
        knn_idx = sorted_idx[1:self.k + 1]     # We exclude x itself (index 0, distance = 0)
        k_dist = distances[sorted_idx[self.k]]  # Distance to the k-th neighbor exactly

        return knn_idx, k_dist, distances

    def _reach_dist(self, x, neighbor, k_dist_neighbor):
        """Computes the reachability distance between x and a neighbor.
        This smooths distances in very dense areas to avoid instability.

        reach_dist = max(k-distance(neighbor), actual distance between x and neighbor)
        """
        d = self._euclidean(x, neighbor)
        return max(k_dist_neighbor, d)

    def _lrd(self, x, X):
        """Computes the Local Reachability Density of point x.
        The higher the density, the closer x is to its neighbors.
        """
       
        knn_idx, k_dist_x, distances = self._k_neighbors(x, X)

        reach_dists = []
        for idx in knn_idx:
            neighbor = X[idx]
            # We need the k-distance of the neighbor to compute reachability distance
            _, k_dist_neighbor, _ = self._k_neighbors(neighbor, X)
            rd = self._reach_dist(x, neighbor, k_dist_neighbor)
            reach_dists.append(rd)

        # Inverse of the average reachability distances = local density
        avg_reach = np.mean(reach_dists)
        return 1.0 / avg_reach if avg_reach > 0 else 0

    def score_samples(self, X):
        """Computes the LOF score for each point in X.
        A score >> 1 means the point is much less dense than its neighbors → outlier.
        A score ≈ 1 means the point has similar density to its neighbors → normal.
        """
        X = np.array(X)
        scores = []

        for x in X:
            knn_idx, _, _ = self._k_neighbors(x, X)

            # Local density of x
            lrd_x = self._lrd(x, X)

            # Mean local density of x's neighbors
            lrd_neighbors = np.mean([self._lrd(X[idx], X) for idx in knn_idx])

            # LOF = ratio of neighbor density to x's density
           
            lof = lrd_neighbors / lrd_x if lrd_x > 0 else float('inf')
            scores.append(lof)

        return np.array(scores)

    def predict(self, X):
        """Predicts whether each point is normal or an outlier.

        Returns:
            0  = normal point
            1 = outlier (anomaly)
        """
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return np.where(scores >= threshold, 1, 0)
    
    def fit_predict(self, X):
        """Fits the model and returns anomaly labels for X.
    
        Returns:
        np.ndarray: 1 = outlier, 0 = normal
        """
        return self.fit(X).predict(X)
    
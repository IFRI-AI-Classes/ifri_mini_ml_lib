import numpy as np
import pandas as pd


class LOF:
    """
    Local Outlier Factor
    Detects outliers by comparing the local density of each point
    to the local density of its neighbors.
    
    By convention ,  0 = normal, 1 = outlier 
    
    """

    def __init__(self, n_neighbors=20, contamination=0.1):
        self.k = n_neighbors            # Number of neighbors to consider
        self.contamination = contamination  # Expected proportion of outliers
        self.X_train = None
        self.distance_matrix_ = None  # pairwise distances (n_train × n_train)
        self.k_distances_     = None  # k-distance of each train point
        self.lrd_train_       = None  # lrd of each train point

    
    def fit(self, X):
        """ Stores training data and precomputes:
          - the full pairwise distance matrix  
          - the k-distance of every train point
          - the local reachability density of every train point

        Precomputing everything here means predict(X_test) only needs
        to compute distances from X_test to X_train, not recompute
        the entire train structure.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            self"""
        
        self.X_train = np.array(X)
        n = len(self.X_train)
        
        #  pairwise distance matrix
        self.distance_matrix_ = self._compute_pairwise_distances(self.X_train)
        
        #k-distance of every train point
        self.k_distances_ = np.array([
            self._k_neighbors_train(i)[1] for i in range(n)
        ])
        
        #lrd of every train point
        self.lrd_train_ = np.array([
            self._lrd_train(i) for i in range(n)
        ])
        
        return self

    def _compute_pairwise_distances(self, X):
        """
        Computes the full pairwise Euclidean distance matrix for X.

        Uses broadcasting to avoid a Python loop:
            diff[i, j] = X[i] - X[j]  → shape (n, n, n_features)
            then sqrt(sum of squares along axis=2)

        Returns:
            matrix of shape (n, n) where entry [i,j] = dist(X[i], X[j])
        """
        # X[:, np.newaxis, :] shape: (n, 1, d)
        # X[np.newaxis, :, :] shape: (1, n, d)
        # diff shape after broadcast: (n, n, d)
        diff    = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        squared = diff ** 2
        return np.sqrt(squared.sum(axis=2))   # shape (n, n)

    def _euclidean(self, x, X):
        """
        Computes Euclidean distance from one  point x to all rows of X.
        Used for X_test points that are not in the distance matrix.

        Returns:
            1D array of shape (n,)
        """
        diff = X - x                          
        return np.sqrt((diff ** 2).sum(axis=1))

    def _k_neighbors_train(self, i):
        """
        Finds the k nearest neighbors of train point i within X_train.

        Reads directly from self.distance_matrix_  no recomputation.
        
        skips index 0 after sorting because the closest point
        to train point i is always itself (distance = 0). This is only
        valid when the evaluated point is part of X_train.

        Args:
            i: integer index of the point in X_train

        Returns:
            knn_idx  : indices of the k nearest neighbors in X_train
            k_dist   : distance to the k-th nearest neighbor
        """
        distances  = self.distance_matrix_[i]       # row i of the matrix
        sorted_idx = np.argsort(distances)

        # sorted_idx[0] == i always (distance = 0) → skip it
        knn_idx = sorted_idx[1:self.k + 1]
        k_dist  = distances[sorted_idx[self.k]]     # k-distance of point i

        return knn_idx, k_dist

    def _k_neighbors_test(self, x):
        """
        Finds the k nearest neighbors of a new test point x within X_train.

        Computes distances on the fly from x to all X_train points.

        does NOT skip index 0 after sorting because x is not
        in X_train ; its minimum distance is not 0, so the closest point
        found at index 0 is a genuine neighbor that must be kept.

        Args:
            x: 1D array of shape (n_features,) — a single test point

        Returns:
            knn_idx  : indices of the k nearest neighbors in X_train
            k_dist   : distance to the k-th nearest neighbor
        """
        distances  = self._euclidean(x, self.X_train)
        sorted_idx = np.argsort(distances)

        # No skip here — x is not in X_train
        knn_idx = sorted_idx[:self.k]
        k_dist  = distances[sorted_idx[self.k - 1]]  # k-distance of test point x

        return knn_idx, k_dist

    def _reach_dist(self, dist_p_o, k_dist_o):
        """
        Reachability distance from point p to neighbor o.

        reach-dist(p, o) = max( k-distance(o), dist(p, o) )

        Smooths distances in dense areas: if p is very close to o,
        we use at least k-distance(o) to avoid instability.

        Args:
            dist_p_o  : actual Euclidean distance between p and o
            k_dist_o  : k-distance of neighbor o (precomputed in fit)

        Returns:
            scalar reachability distance
        """
        return max(k_dist_o, dist_p_o)

    def _lrd_train(self, i):
        """
        Computes the Local Reachability Density of train point i.

        lrd(i) = 1 / mean( reach-dist(i, o) for o in neighbors of i )

        Reads all distances from self.distance_matrix_ .
        Uses self.k_distances_ for the k-distances of neighbors.

        High lrd include that  point i is in a dense area
        Low  lrd include that point i is in a sparse area

        Args:
            i: integer index of the point in X_train

        Returns:
            scalar lrd value
        """
        knn_idx, _ = self._k_neighbors_train(i)

        reach_dists = [
            self._reach_dist(
                dist_p_o = self.distance_matrix_[i, o],  # dist(i, neighbor o)
                k_dist_o = self.k_distances_[o]          # k-distance of neighbor o
            )
            for o in knn_idx
        ]

        avg_reach = np.mean(reach_dists)
        return 1.0 / avg_reach if avg_reach > 0 else 0.0

    def _lrd_test(self, x):
        """
        Computes the Local Reachability Density of a new test point x.

        Same formula as _lrd_train() but x is not in X_train, so:
          - neighbors are found via _k_neighbors_test() (no index skip)
          - distances from x to neighbors are computed on the fly
          - k-distances of neighbors are read from self.k_distances_ (precomputed)

        Args:
            x: 1D array of shape (n_features,) — a single test point

        Returns:
            scalar lrd value
        """
        knn_idx, _ = self._k_neighbors_test(x)

        # Distances from x to each of its k neighbors in X_train
        distances_x = self._euclidean(x, self.X_train)

        reach_dists = [
            self._reach_dist(
                dist_p_o = distances_x[o],        # dist(x, neighbor o)
                k_dist_o = self.k_distances_[o]   # k-distance of o (from fit)
            )
            for o in knn_idx
        ]

        avg_reach = np.mean(reach_dists)
        return 1.0 / avg_reach if avg_reach > 0 else 0.0

    def score_samples(self, X):
        """Computes the LOF score for each point in X.
        A score >> 1 means the point is much less dense than its neighbors → outlier.
        A score ≈ 1 means the point has similar density to its neighbors → normal.
        """
        X = np.array(X)
        scores = []

        for x in X:
            knn_idx, _, _ = self._k_neighbors_test(x, X)

            # Local density of x
            lrd_x = self._lrd_test(x, X)

            # Mean local density of x's neighbors
            lrd_neighbors = np.mean([self.lrd_train_[o] for o in knn_idx])

            # LOF = ratio of neighbor density to x's density
           
            lof = lrd_neighbors / lrd_x if lrd_x > 0 else float('inf')
            scores.append(lof)

        return np.array(scores)

    def predict(self, X):
        """Predicts whether each point is normal or an outlier.
        Threshold = (1 - contamination) percentile of LOF scores.
        Points above the threshold are flagged as outliers.

        Returns:
            1D int array: 1 = outlier, 0 = normal
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
import numpy as np


class DecisionTreeRegressor:
    """
    A decision tree regressor.

    Args:
        max_depth (int, optional): Maximum depth of the tree. Defaults to None.
        min_samples_split (int): Minimum samples required to split a node. Defaults to 2.
        min_samples_leaf (int): Minimum samples required at a leaf node. Defaults to 1.
        min_impurity_decrease (float): Minimum variance reduction for a split. Defaults to 0.0.
        max_features (int or None): Number of features to consider per split. Defaults to None.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y, depth=0):
        n_samples = X.shape[0]

        # Stop conditions
        if (n_samples < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth) or
            len(np.unique(y)) == 1):
            self.tree = self._leaf_value(y)
            return self.tree

        # Best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain < self.min_impurity_decrease:
            self.tree = self._leaf_value(y)
            return self.tree

        left_mask  = X[:, best_feature] < best_threshold
        right_mask = ~left_mask

        if (np.sum(left_mask) < self.min_samples_leaf or
            np.sum(right_mask) < self.min_samples_leaf):
            self.tree = self._leaf_value(y)
            return self.tree

        self.tree = {
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left":  self.fit(X[left_mask],  y[left_mask],  depth + 1),
            "right": self.fit(X[right_mask], y[right_mask], depth + 1)
        }
        return self.tree

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feature, best_threshold = None, None

        # Random feature selection (for Random Forest)
        n_features = X.shape[1]
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features,
                                               min(self.max_features, n_features),
                                               replace=False)
        else:
            feature_indices = range(n_features)

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feature_index], [25, 50, 75])

            for threshold in thresholds:
                left_mask  = X[:, feature_index] < threshold
                right_mask = ~left_mask

                if (np.sum(left_mask) < self.min_samples_leaf or
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue

                gain = self._variance_reduction(y, left_mask, right_mask)
                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _mse(self, y):
        """Calculate the MSE of a node."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _variance_reduction(self, y, left_mask, right_mask):
        """Calculate the variance reduction of a split."""
        n = len(y)
        n_left  = np.sum(left_mask)
        n_right = np.sum(right_mask)
        return self._mse(y) - (n_left/n  * self._mse(y[left_mask]) +
                                n_right/n * self._mse(y[right_mask]))

    def _leaf_value(self, y):
        """Returns the average of the y — values ​​from the leaf."""
        return np.mean(y)

    def predict(self, X):
        if self.tree is None:
            raise ValueError("The model needs to be trained before it can make predictions.")
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x, tree=None):
        if tree is None:
            tree = self.tree
        if isinstance(tree, dict):
            if x[tree["feature_index"]] < tree["threshold"]:
                return self._predict_single(x, tree["left"])
            else:
                return self._predict_single(x, tree["right"])
        return tree

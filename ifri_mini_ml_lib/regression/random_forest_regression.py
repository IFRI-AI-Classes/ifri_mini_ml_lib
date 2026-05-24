import numpy as np
from.decision_tree_regressor import DecisionTreeRegressor

class RandomForestRegressor:
    """
    Simple Random Forest Regressor.

    Args:
        n_estimators (int): Number of trees.
        max_depth (int): Maximum tree depth.
        max_features (int or None): Number of features per split.
    """

    def __init__(self, n_estimators=10, max_depth=5, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        Generate a bootstrap sample.

        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): Target values.

        Returns:
            tuple: Sampled X and y.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Train the model.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        """
        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Make predictions.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted values.
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)
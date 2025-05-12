import numpy as np

class KNN:
    """
    Description:
        A simple implementation of the K-Nearest Neighbors (KNN) algorithm 
        for classification and regression tasks.

    Args:
        k (int): Number of nearest neighbors to consider (default is 3).
        task (str): Type of task - either 'classification' or 'regression' (default is 'classification').

    Example:
        >>> knn = KNN(k=5, task='classification')
        >>> knn.fit(X_train, y_train)
        >>> predictions = knn.predict(X_test)
    """
    def __init__(self, k=3, task='classification'):
        self.k = k
        self.task = task 

    def fit(self, X, y):
        """
        Description:
            Stores the training data.

        Args:
            X (array-like): Feature vectors of the training data.
            y (array-like): Target labels or values for the training data.
        """
        self.x_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Description:
            Predicts the labels or values for a set of input samples.

        Args:
            X (array-like): Feature vectors of the test data.

        Returns:
            list: List of predicted labels or values.
        """
        return [self._predict(x) for x in X]

    def _predict(self, x):
        """
        Description:
            Predicts the label or value for a single input sample using the KNN algorithm.

        Args:
            x (array-like): A single input sample.

        Returns:
            Predicted label (for classification) or numerical value (for regression).
        """
        # Compute distances between x and all examples in the training set
        distances = np.linalg.norm(self.x_train - x, axis=1)

        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Retrieve the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        if self.task == 'regression':
            # Return the average value for regression
            return np.mean(k_nearest_labels)
        else:
            # Count the occurrences of each label
            label_counts = {}
            for label in k_nearest_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            # Return the label with the highest count
            return max(label_counts, key=label_counts.get)

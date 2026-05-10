"""
Linear Support Vector Machine (SVM) module.

This module provides a LinearSVM class for binary and multiclass classification
using gradient descent optimization with hinge loss.

Example:
    >>> from svm_linear import LinearSVM
    >>> model = LinearSVM()
    >>> model.fit([[1, 2], [2, 3], [5, 5], [6, 6]], [-1, -1, 1, 1])
    >>> model.predict([[4, 4]])
    [1]
"""

from typing import List, Union

import numpy as np

__all__ = ["LinearSVM"]


class LinearSVM:
    """
    Linear Support Vector Machine (SVM) classifier.

    Implements binary and multiclass classification using gradient descent
    on the primal SVM objective with hinge loss. Multiclass problems are
    handled with a One-vs-Rest (OvR) strategy.

    The primal optimization problem being solved is:

        min  (1/2)||w||^2 + C * sum(max(0, 1 - y_i(w^T x_i + b)))
         w,b

    where C controls the trade-off between margin maximization and
    training error tolerance.

    Args:
        learning_rate (float): Step size for gradient descent (default: 0.001).
        C (float): Regularization parameter. Smaller values = stronger
            regularization. Larger values = fewer misclassifications
            tolerated (default: 1.0).
        n_iters (int): Number of gradient descent iterations (default: 1000).

    Example:
        >>> model = LinearSVM(learning_rate=0.01, C=1.0, n_iters=1000)
        >>> model.fit([[1, 2], [2, 3], [5, 5], [6, 6]], [-1, -1, 1, 1])
        >>> model.predict([[4, 4]])
        [1]
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        C: float = 1.0,
        n_iters: int = 1000,
    ) -> None:
        self.learning_rate = learning_rate
        self.C = C
        self.n_iters = n_iters

        self.w = None           # weight vector (binary) or list of vectors (OvR)
        self.b = None           # bias term (binary) or list of floats (OvR)
        self.classes_ = None    # unique class labels found during fit

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _fit_binary(self, X: np.ndarray, y: np.ndarray):
        """
        Train a single binary SVM for labels in {-1, +1}.

        Applies gradient descent on the hinge loss. For each sample:
        - If y_i(w^T x_i + b) >= 1  -> only apply weight decay (regularization)
        - Otherwise                  -> apply hinge loss gradient update

        Args:
            X (np.ndarray): Training data of shape [n_samples, n_features].
            y (np.ndarray): Binary labels of shape [n_samples], values in {-1, +1}.

        Returns:
            tuple: (weights, bias) — the trained parameters.

        Example:
            >>> w, b = model._fit_binary(np.array([[1,2],[3,4]]), np.array([-1, 1]))
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Functional margin for this sample
                margin = y[idx] * (np.dot(x_i, w) + b)

                if margin >= 1:
                    # Sample correctly classified outside margin: only regularize
                    w -= self.learning_rate * w
                else:
                    # Margin violated: hinge loss gradient update
                    w -= self.learning_rate * (w - self.C * y[idx] * x_i)
                    b += self.learning_rate * self.C * y[idx]

        return w, b

    def _decision_scores(
        self, X: np.ndarray, w: np.ndarray, b: float
    ) -> np.ndarray:
        """
        Compute raw decision scores (signed distances to the hyperplane).

        Args:
            X (np.ndarray): Input data of shape [n_samples, n_features].
            w (np.ndarray): Weight vector of shape [n_features].
            b (float): Bias term.

        Returns:
            np.ndarray: Decision scores of shape [n_samples].

        Example:
            >>> scores = model._decision_scores(X, w, b)
        """
        return np.dot(X, w) + b

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[List, np.ndarray],
        y: Union[List, np.ndarray],
    ) -> "LinearSVM":
        """
        Train the Linear SVM model.

        Automatically detects binary vs. multiclass problems:
        - Binary     : one classifier, labels mapped to {-1, +1}.
        - Multiclass : one classifier per class using One-vs-Rest (OvR).

        Args:
            X (Union[List, np.ndarray]): Training data of shape [n_samples, n_features].
            y (Union[List, np.ndarray]): Target labels of shape [n_samples].
                Can be integers, strings, or any comparable type.

        Returns:
            LinearSVM: The fitted model instance (allows method chaining).

        Raises:
            ValueError: If X or y are empty, or have incompatible shapes.

        Example:
            >>> model = LinearSVM()
            >>> model.fit([[1, 2], [2, 3], [5, 5], [6, 6]], [-1, -1, 1, 1])
        """
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y)

        # Input validation
        if X_arr.size == 0 or y_arr.size == 0:
            raise ValueError("X and y cannot be empty.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got X: {X_arr.shape[0]}, y: {y_arr.shape[0]}."
            )

        self.classes_ = np.unique(y_arr)

        if len(self.classes_) == 2:
            # --- Binary case ---
            # Map original labels to {-1, +1}
            y_binary = np.where(y_arr == self.classes_[1], 1, -1).astype(float)
            self.w, self.b = self._fit_binary(X_arr, y_binary)

        else:
            # --- Multiclass case: One-vs-Rest ---
            self.w = []
            self.b = []
            for cls in self.classes_:
                y_binary = np.where(y_arr == cls, 1, -1).astype(float)
                w, b = self._fit_binary(X_arr, y_binary)
                self.w.append(w)
                self.b.append(b)

        return self

    def predict(self, X: Union[List, np.ndarray]) -> List:
        """
        Predict class labels for input samples.

        Args:
            X (Union[List, np.ndarray]): Input data of shape [n_samples, n_features].

        Returns:
            list: Predicted class labels of shape [n_samples].

        Raises:
            ValueError: If X is empty.
            RuntimeError: If the model has not been fitted yet.

        Example:
            >>> model.predict([[4, 4]])
            [1]
        """
        if self.w is None:
            raise RuntimeError(
                "Model is not fitted yet. Call 'fit' before 'predict'."
            )
        if X is None or len(X) == 0:
            raise ValueError("Input data for prediction cannot be empty.")

        X_arr = np.array(X, dtype=float)

        if len(self.classes_) == 2:
            # Binary: sign of the decision score selects the class
            scores = self._decision_scores(X_arr, self.w, self.b)
            predictions = np.where(scores >= 0, self.classes_[1], self.classes_[0])
        else:
            # Multiclass OvR: class with the highest decision score wins
            all_scores = np.column_stack([
                self._decision_scores(X_arr, w, b)
                for w, b in zip(self.w, self.b)
            ])
            predictions = self.classes_[np.argmax(all_scores, axis=1)]

        return predictions.tolist()

    def score(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> float:
        """
        Compute the accuracy of the model on the given test data.

        Args:
            X (Union[List, np.ndarray]): Test data of shape [n_samples, n_features].
            y (Union[List, np.ndarray]): True labels of shape [n_samples].

        Returns:
            float: Accuracy score between 0.0 and 1.0.

        Raises:
            RuntimeError: If the model has not been fitted yet.

        Example:
            >>> model.score([[1, 2], [5, 5]], [-1, 1])
            1.0
        """
        y_arr = np.array(y)
        predictions = np.array(self.predict(X))
        return float(np.mean(predictions == y_arr))

    def __repr__(self) -> str:
        return (
            f"LinearSVM(learning_rate={self.learning_rate}, "
            f"C={self.C}, n_iters={self.n_iters})"
        )
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

class LinearSolver:
    """
    Optimization Solver for Linear SVM with Pegasos Algorithm.

    Description:
        Implements the Pegasos algorithm which solves the primal optimization problem
        of linear soft-margin SVM via stochastic sub-gradient descent.
        
    Args:
        C (float):
            Regularization parameter (inverse of λ = 1/C).
            Controls the bias-variance trade-off:
            - High C   → weak regularization, smaller margin, less bias
            - Low C    → strong regularization, larger margin, more bias
            Default: 1.0

        max_iter (int):
            Number of passes over the data (epochs).
            Default: 1000

        tol (float):
            Convergence tolerance. Training stops if the variation of the loss
            function between two epochs is less than tol.
            Default: 1e-4

        random_state (int or None):
            Seed for the random number generator for reproducibility.
            Default: None

    Attributes:
        w_ (np.ndarray):
            Weight vector of shape (n_features,)

        b_ (float):
            Bias term

        loss_history_ (list of float):
            History of the hinge loss function at each epoch.

    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 tol: float = 1e-4, random_state=None):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.w_ = None
        self.b_ = None
        self.loss_history_ = []

    def solve(self, X: np.ndarray, y: np.ndarray):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        lam = 1.0 / self.C
        prev_loss = float('inf')

        w = np.zeros(n_features)
        b = 0.0
        loss_history = []

        for epoch in range(1, self.max_iter + 1):
            indices = np.random.permutation(n_samples)

            for t, i in enumerate(indices, start=1):
                t_global = (epoch - 1) * n_samples + t
                eta = 1.0 / (lam * t_global)

                margin = y[i] * (np.dot(w, X[i]) + b)

                if margin < 1:
                    w = (1 - eta * lam) * w + eta * y[i] * X[i]
                    b += eta * y[i]
                else:
                    w = (1 - eta * lam) * w

                norm_w = np.linalg.norm(w)
                proj_radius = 1.0 / np.sqrt(lam)
                if norm_w > proj_radius:
                    w *= proj_radius / norm_w

            epoch_loss = self._hinge_loss(X, y, w, b, lam)
            loss_history.append(epoch_loss)

            if abs(prev_loss - epoch_loss) < self.tol:
                break
            prev_loss = epoch_loss

        self.w_ = w
        self.b_ = b
        self.loss_history_ = loss_history
        return w, b

    def _hinge_loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, 
                    b: float, lam: float) -> float:
        margins = y * (X.dot(w) + b)
        hinge = np.maximum(0, 1 - margins)
        regularization = (lam / 2.0) * np.dot(w, w)
        return float(regularization + np.mean(hinge))

class LinearSVM:
    """
    Linear Support Vector Machine (SVM) classifier.

    Implements binary and multiclass classification using the Pegasos algorithm
    for optimization on the primal SVM objective with hinge loss. Multiclass problems are
    handled with a One-vs-Rest (OvR) strategy.

    The primal optimization problem being solved is:

        min  (1/2)||w||^2 + C * sum(max(0, 1 - y_i(w^T x_i + b)))
         w,b

    where C controls the trade-off between margin maximization and
    training error tolerance.

    Args:
        C (float): Regularization parameter. Smaller values = stronger
            regularization. Larger values = fewer misclassifications
            tolerated (default: 1.0).
        n_iters (int): Number of Pegasos iterations (default: 1000).
        tol (float): Convergence tolerance (default: 1e-4).
        random_state (int or None): Random seed for reproducibility (default: None).

    Example:
        >>> model = LinearSVM(C=1.0, n_iters=1000)
        >>> model.fit([[1, 2], [2, 3], [5, 5], [6, 6]], [-1, -1, 1, 1])
        >>> model.predict([[4, 4]])
        [1]
    """

    def __init__(
        self,
        C: float = 1.0,
        n_iters: int = 1000,
        tol: float = 1e-4,
        random_state = None,
    ) -> None:
        self.C = C
        self.n_iters = n_iters
        self.tol = tol
        self.random_state = random_state

        self.w = None           # weight vector (binary) or list of vectors (OvR)
        self.b = None           # bias term (binary) or list of floats (OvR)
        self.classes_ = None    # unique class labels found during fit

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _fit_binary(self, X: np.ndarray, y: np.ndarray):
        """
        Train a single binary SVM for labels in {-1, +1}.

        Uses the Pegasos algorithm for optimization.

        Args:
            X (np.ndarray): Training data of shape [n_samples, n_features].
            y (np.ndarray): Binary labels of shape [n_samples], values in {-1, +1}.

        Returns:
            tuple: (weights, bias) — the trained parameters.

        """
        solver = LinearSolver(C=self.C, max_iter=self.n_iters, tol=self.tol, random_state=self.random_state)
        w, b = solver.solve(X, y)
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
            y (v&Union[List, np.ndarray]): True labels of shape [n_samples].

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
            f"LinearSVM(C={self.C}, n_iters={self.n_iters}, tol={self.tol}, random_state={self.random_state})"
        )
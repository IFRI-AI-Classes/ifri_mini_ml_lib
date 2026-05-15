"""
Support Vector Machine (SVM) Module.

This unified module provides:
    - BaseSVM: Abstract base class for all SVM models
    - SVMLinear: Linear SVM with Pegasos optimization (binary and multiclass)
    - SVMRBF: Binary RBF SVM with SMO optimization
    - SVMRBFOvO: Multiclass RBF SVM with One-vs-One strategy

References:
    - Cortes, C., & Vapnik, V. (1995). Support-vector networks.
      Machine Learning, 20(3), 273–297.
    - Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn,
      Keras & TensorFlow (2nd ed.). O'Reilly Media. Chapter 5.
    - scikit-learn API Reference: https://scikit-learn.org/stable/modules/svm.html
"""

from abc import ABC, abstractmethod
from itertools import combinations
from typing import List, Union

import numpy as np

__all__ = [
    "BaseSVM",
    "SVMLinear",
    "SVMRBF",
    "SVMRBFOvO",
    "rbf_kernel",
    "smo",
    # popular aliases
    "SVMLinear",
    "SVMRBF",
]


# ─── Abstract Base Class ───────────────────────────────────────────────


class BaseSVM(ABC):
    """
    Abstract base class for all SVM models in the library.

    Description:
        Defines the common interface (contract) that every SVM model must respect.
        Concrete subclasses (SVMLinear, SVMRBF) must implement the `fit` and
        `predict` methods. The `score` method is provided by default and can
        be overridden if needed.

        Design pattern used: "Template Method" — the high-level structure
        is fixed here, details are delegated to subclasses.

    Attributes:
        is_fitted_ (bool): Indicates whether the model has been trained.

    Examples:
        This class cannot be instantiated directly:

        >>> svm = BaseSVM()  # Raises TypeError
        TypeError: Can't instantiate abstract class BaseSVM

        Correct usage via a subclass:

        >>> svm = SVMLinear(C=1.0)
        >>> svm.fit(X_train, y_train)
        >>> svm.predict(X_test)

    References:
        - scikit-learn BaseEstimator:
          https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py
    """

    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X, y):
        """
        Trains the SVM model on the provided data.

        Description:
            Abstract method — each subclass implements its own training
            procedure (linear optimization, SMO, etc.).

        Args:
            X (array-like of shape (n_samples, n_features)):
                Training data matrix. Each row is an example,
                each column is a feature.
            y (array-like of shape (n_samples,)):
                Target labels vector. Expected values are {-1, +1}
                for binary SVM classification. Labels {0, 1} are
                automatically converted.

        Returns:
            self: The trained instance (enables method chaining svm.fit().predict()).

        Raises:
            NotImplementedError: If the subclass does not implement fit().
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts class labels for new data.

        Description:
            Abstract method — each subclass implements its own decision
            rule according to the kernel used.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Data for which to make predictions.

        Returns:
            np.ndarray of shape (n_samples,):
                Predicted labels. Values in {-1, +1}.

        Raises:
            AttributeError: If the model has not yet been trained.
        """
        pass

    def score(self, X, y):
        """
        Computes the accuracy of the model on a dataset.

        Description:
            Utility method provided by the base class. Compares the model's
            predictions to the true labels and returns the percentage of
            correct predictions.

            This method can be overridden by subclasses if a different
            metric is desired.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Test data.
            y (array-like of shape (n_samples,)):
                True labels ({-1, +1} or {0, 1}).

        Returns:
            float: Accuracy between 0.0 (0%) and 1.0 (100%).

        Examples:
            >>> svm = SVMLinear(C=1.0)
            >>> svm.fit(X_train, y_train)
            >>> acc = svm.score(X_test, y_test)
            >>> print(f"Accuracy: {acc:.2%}")
            Accuracy: 95.00%
        """
        y = np.array(y)
        # Normalize labels {0,1} → {-1,+1} for comparison
        if set(np.unique(y)).issubset({0, 1}):
            y = np.where(y == 0, -1, 1)
        predictions = np.array(self.predict(X))
        return float(np.mean(predictions == y))

    def _check_is_fitted(self):
        """
        Verifies that the model has been trained before making predictions.

        Description:
            Internal utility method called in `predict` by subclasses to raise
            an explicit error if `fit` has not yet been called.

        Raises:
            AttributeError: Clear message indicating that fit() is required.
        """
        if not self.is_fitted_:
            raise AttributeError(
                f"This {type(self).__name__} model is not yet trained. "
                "Call fit(X, y) before predict()."
            )


# ─── Linear SVM with Pegasos ───────────────────────────────────────────


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


class SVMLinear(BaseSVM):
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
        >>> model = SVMLinear(C=1.0, n_iters=1000)
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
        super().__init__()
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
    ) -> "SVMLinear":
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
            SVMLinear: The fitted model instance (allows method chaining).

        Raises:
            ValueError: If X or y are empty, or have incompatible shapes.

        Example:
            >>> model = SVMLinear()
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

        # mark fitted
        self.is_fitted_ = True
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
        # ensure model was fitted
        self._check_is_fitted()
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
        # use BaseSVM.score to keep behaviour consistent and label normalization
        return super().score(X, y)

    def __repr__(self) -> str:
        return (
            f"SVMLinear(C={self.C}, n_iters={self.n_iters}, tol={self.tol}, random_state={self.random_state})"
        )


# ─── RBF Kernel SVM with SMO ───────────────────────────────────────────


def rbf_kernel(X, Y, gamma=1.0):
    """Compute the pairwise RBF kernel matrix between two sample sets.

    Args:
        X (np.ndarray): First matrix of shape (n_samples_X, n_features).
        Y (np.ndarray): Second matrix of shape (n_samples_Y, n_features).
        gamma (float): RBF width parameter. Larger values make the kernel
            decay faster with distance.

    Returns:
        np.ndarray: Kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    diff = X[:, np.newaxis, :] - Y  # Broadcasting builds all pairwise differences without explicit Python loops. (n,m,d)
    sq_distances = np.sum(diff**2, axis=2)  # Squared Euclidean distance is the quantity used by the RBF kernel. (n,m)
    return np.exp(-gamma * sq_distances)


def smo(K, y, C, tol=1e-3, max_iter=100):
    """Solve the SVM dual problem with a basic SMO loop.

    Args:
        K (np.ndarray): Gram matrix of shape (n_samples, n_samples).
        y (np.ndarray): Binary labels encoded as -1 and +1.
        C (float): Regularization strength.
        tol (float): KKT violation tolerance.
        max_iter (int): Maximum number of outer iterations.

    Returns:
        tuple[np.ndarray, float]: The optimized alpha coefficients and bias.
    """
    n = len(y)
    alphas = np.zeros(n)
    b = 0
    
    for _ in range(max_iter):
        errors = (alphas * y) @ K + b - y  # Current dual residuals for every training sample.
        
        vi = y * errors
        violations = np.concatenate([
            np.where((vi < -tol) & (alphas < C))[0],
            np.where((vi > tol) & (alphas > 0))[0]
        ])
        
        if len(violations) == 0:
            break
            
        i = violations[np.argmax(np.abs(vi[violations]))]  # Pick the worst violator so each step focuses on the largest error.
        
        diff_errors = np.abs(errors[i] - errors)
        diff_errors[i] = 0
        j = np.argmax(diff_errors)  # Choose a second point with the most different prediction error.
        
        eta = K[i,i] + K[j,j] - 2*K[i,j]
        if eta <= 0:
            continue
            
        if y[i] == y[j]:
            L = max(0, alphas[i] + alphas[j] - C)
            H = min(C, alphas[i] + alphas[j])
        else:
            L = max(0, alphas[j] - alphas[i])
            H = min(C, C + alphas[j] - alphas[i])
            
        if L == H:
            continue
            
        alpha_j_new = alphas[j] + y[j] * (errors[i] - errors[j]) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)  # Clip the updated coefficient so it stays inside the feasible box.
        alpha_i_new = alphas[i] + y[i] * y[j] * (alphas[j] - alpha_j_new)  # Recover the paired coefficient from the equality constraint.
        
        b1 = b - errors[i] - y[i]*(alpha_i_new - alphas[i])*K[i,i] - y[j]*(alpha_j_new - alphas[j])*K[i,j]
        b2 = b - errors[j] - y[i]*(alpha_i_new - alphas[i])*K[i,j] - y[j]*(alpha_j_new - alphas[j])*K[j,j]
        
        if 0 < alpha_i_new < C:
            b = b1
        elif 0 < alpha_j_new < C:
            b = b2
        else:
            b = (b1 + b2) / 2
            
        alphas[i] = alpha_i_new
        alphas[j] = alpha_j_new
    
    return alphas, b


class SVMRBF(BaseSVM):
    """Binary SVM classifier trained with an RBF kernel and SMO."""

    def __init__(self, C=1.0, gamma=1.0):
        """Initialize the classifier.

        Args:
            C (float): Regularization parameter.
            gamma (float): RBF kernel parameter.
        """
        super().__init__()
        self.C = C
        self.gamma = gamma
        self.support_vectors = None
        self.support_alphas = None
        self.support_labels = None
        self.b = None

    def fit(self, X, y):
        """Fit the classifier and keep only support vectors."""
        K = rbf_kernel(X, X, self.gamma)
        alphas, b = smo(K, y, self.C)
        support_indices = np.where(alphas > 0)[0]
        self.support_vectors = X[support_indices]
        self.support_alphas = alphas[support_indices]
        self.support_labels = y[support_indices]
        self.b = b
        self.is_fitted_ = True

    def predict(self, X):
        """Predict class labels for a batch of samples."""
        self._check_is_fitted()
        K = rbf_kernel(self.support_vectors, X, self.gamma)
        scores = (self.support_alphas * self.support_labels) @ K + self.b
        return np.sign(scores)


class SVMRBFOvO(BaseSVM):
    """One-vs-one multi-class wrapper around the binary RBF SVM."""

    def __init__(self, C=1.0, gamma=1.0):
        """Initialize the multi-class wrapper.

        Args:
            C (float): Regularization parameter for each binary classifier.
            gamma (float): RBF kernel parameter.
        """
        super().__init__()
        self.C = C
        self.gamma = gamma
        self.classifiers = {}
    
    def fit(self, X, y):
        """Train one binary classifier for each pair of classes."""
        classes = np.unique(y)
        pairs = list(combinations(classes, 2))
        
        for (c1, c2) in pairs:
            mask = (y == c1) | (y == c2)
            X_pair = X[mask]
            y_binary = np.where(y[mask] == c1, 1, -1)
            
            clf = SVMRBF(C=self.C, gamma=self.gamma)
            clf.fit(X_pair, y_binary)
            self.classifiers[(c1, c2)] = clf
        self.is_fitted_ = True
    
    def predict_one(self, x):
        """Predict a single sample by majority vote."""
        votes = {}
        for (c1, c2), clf in self.classifiers.items():
            prediction = clf.predict(x.reshape(1, -1))[0]
            if prediction == 1:
                votes[c1] = votes.get(c1, 0) + 1
            else:
                votes[c2] = votes.get(c2, 0) + 1
        return max(votes, key=votes.get)
    
    def predict(self, X):
        """Predict class labels for a batch of samples."""
        self._check_is_fitted()
        return np.array([self.predict_one(x) for x in X])



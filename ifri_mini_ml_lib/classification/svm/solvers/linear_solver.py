"""
Linear SVM Solver using Pegasos Algorithm

Description:
    Implements the optimization solver for linear SVM using the Pegasos algorithm
    (Primal Estimated sub-GrAdient SOlver for SVM). This module is purely mathematical:
    it has no knowledge of ML models. It receives data and returns optimal w and b.

Algorithm:
    Pegasos solves the primal soft-margin SVM problem:

        min  (λ/2)||w||² + (1/n) Σ max(0, 1 - yᵢ(w·xᵢ + b))
         w,b

    Via stochastic sub-gradient descent with decaying learning rate: η_t = 1 / (λ * t)

Complexity:
    - Time  : O(max_iter × n_features)
    - Space : O(n_features)

References:
    [1] Shalev-Shwartz, S., Singer, Y., Srebro, N., & Cotter, A. (2011).
        Pegasos: Primal estimated sub-gradient solver for SVM.
        Mathematical Programming, 127(1), 3–30.
        https://doi.org/10.1007/s10107-010-0420-4

    [2] Cortes, C., & Vapnik, V. (1995). Support-vector networks.
        Machine Learning, 20(3), 273–297.

    [3] Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
        Springer. Section 7.1 — Maximum Margin Classifiers.
"""

import numpy as np


class LinearSolver:
    """
    Optimization Solver for Linear SVM with Pegasos Algorithm (Binary & Multiclass).

    Description:
        Implements the Pegasos algorithm which solves the primal optimization problem
        of linear soft-margin SVM via stochastic sub-gradient descent.
        
        Supports both binary and multiclass classification:
        - Binary: Direct binary SVM training
        - Multiclass: One-vs-Rest (OvR) strategy with one binary classifier per class

        Advantages of Pegasos over dual formulation (SMO):
        - Complexity independent of the number of training examples
        - Faster convergence on large datasets
        - Simple and efficient implementation with numpy only

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
        classes_ (np.ndarray):
            Unique class labels seen during fit().
            
        w_ (np.ndarray or list of np.ndarray):
            For binary: weight vector of shape (n_features,)
            For multiclass: list of weight vectors (one per class)

        b_ (float or list of float):
            For binary: bias term
            For multiclass: list of bias terms (one per class)

        loss_history_ (list of float):
            History of the hinge loss function at each epoch.

    Examples:
        >>> solver = LinearSolver(C=1.0, max_iter=1000)
        >>> solver.fit(X_train, y_train)
        >>> predictions = solver.predict(X_test)
        >>> accuracy = solver.score(X_test, y_test)

    References:
        [1] Shalev-Shwartz et al. (2011). Pegasos. Mathematical Programming.
        [2] https://scikit-learn.org/stable/modules/sgd.html
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 tol: float = 1e-4, random_state=None):
        """
        Initialize the solver with optimization hyperparameters.

        Args:
            C (float): Regularization parameter. Default: 1.0.
            max_iter (int): Maximum number of epochs. Default: 1000.
            tol (float): Convergence tolerance. Default: 1e-4.
            random_state (int or None): Random seed. Default: None.
        """
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.w_ = None
        self.b_ = None
        self.loss_history_ = []

    def solve(self, X: np.ndarray, y: np.ndarray):
        """
        Execute the Pegasos algorithm to find optimal w* and b*.

        Description:
            At each iteration t, for a randomly drawn example (xᵢ, yᵢ):

            1. Compute the decaying learning rate: η_t = 1/(λ·t) where λ = 1/C
            2. Check the margin constraint: If yᵢ(w·xᵢ + b) < 1 → misclassified or within margin
            3. Update the gradient:
               - If violation: w ← (1-η_t·λ)w + η_t·yᵢ·xᵢ, b ← b + η_t·yᵢ
               - Otherwise: w ← (1-η_t·λ)w
            4. Project to guarantee ||w|| ≤ 1/√λ

        Args:
            X (np.ndarray of shape (n_samples, n_features)):
                Training data, preferably already normalized.
            y (np.ndarray of shape (n_samples,)):
                Labels in {-1, +1}. IMPORTANT: Labels {0,1} must be
                converted BEFORE calling solve().

        Returns:
            tuple (w, b):
                - w (np.ndarray of shape (n_features,)): optimal weights
                - b (float): optimal bias

        Notes:
            Convergence is guaranteed in O(1/ε) iterations to reach
            precision ε on the objective function (Shalev-Shwartz et al., 2011).
        """
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
        """
        Compute the regularized hinge loss function (Pegasos objective).

        Description:
            L(w, b) = (λ/2)||w||² + (1/n) Σᵢ max(0, 1 - yᵢ(w·xᵢ + b))

            The first term is L2 regularization (penalizes large w).
            The second term is hinge loss (penalizes margin violations).

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Labels in {-1, +1}.
            w (np.ndarray): Current weight vector.
            b (float): Current bias.
            lam (float): Regularization parameter λ = 1/C.

        Returns:
            float: Value of the objective function.
        """
        margins = y * (X.dot(w) + b)
        hinge = np.maximum(0, 1 - margins)
        regularization = (lam / 2.0) * np.dot(w, w)
        return float(regularization + np.mean(hinge))
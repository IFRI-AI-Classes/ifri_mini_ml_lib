"""
Support Vector Machine (SVM) Module.

References:
    - Cortes, C., & Vapnik, V. (1995). Support-vector networks.
      Machine Learning, 20(3), 273–297.
    - Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn,
      Keras & TensorFlow (2nd ed.). O'Reilly Media. Chapter 5.
    - scikit-learn API Reference: https://scikit-learn.org/stable/modules/svm.html
"""

import numpy as np

from ifri_mini_ml_lib.utils.svm_functions_interns import (
    rbf_kernel, smo, hinge_loss, pegasos, 
    fit_linear, fit_rbf, predict_linear, predict_rbf
)

class SVM:
    """
    Unified SVM (Support Vector Machine) classifier.

    Description:
        The Support Vector Machine (SVM) is a classification algorithm that seeks the best 
        possible boundary to separate data into classes. Depending on the nature of the data, 
        two approaches are available: if the data can be separated by a line or a hyperplane,
        the linear kernel optimized by the Pegasos algorithm is used. If the data is more complex 
        and cannot be separated directly, the Restricted Boundary Function (RBF) kernel is used,
        which implicitly projects the data into a higher-dimensional space where it becomes separable,
        optimized by the Selective Mapping Organization (SMO) algorithm. In both cases, when 
        there are more than two classes to separate, the linear kernel adopts a One-vs-Rest strategy,
        resulting in one classifier per class, while the RBF kernel adopts a One-vs-One strategy, 
        resulting in one classifier for each possible pair of classes.

    Args:
        kernel (str): Kernel type: 'linear' (Pegasos) or 'rbf' (SMO). Default: 'linear'.
        C (float): Regularization parameter > 0. Margin vs error tradeoff. Default: 1.0.
        gamma (float): RBF kernel parameter. Controls boundary curvature. Default: 1.0.
        max_iter (int): Maximum optimization iterations. Default: 1000.
        tol (float): Convergence tolerance. Default: 1e-4.
        random_state (int or None): Random seed for reproducibility. Default: None.

    Attributes:
        classes_ (np.ndarray): Unique classes detected during fit().
        is_fitted_ (bool): Whether the model has been trained.
        w_ (np.ndarray or list): Weight vector(s) learned. List if multiclass.
        b_ (float or list): Bias learned. List if multiclass.
        loss_history_ (list): Hinge loss history per epoch (linear kernel only).
        support_vectors_ (np.ndarray): Support vectors identified during training(RBF kernel only).
        support_alphas_ (np.ndarray): Lagrange multipliers (RBF kernel only).
        support_labels_ (np.ndarray): Labels of support vectors (RBF kernel only).

    Examples:
        Classification with linear kernel:       
        >>> model = SVM(kernel='linear', C=1.0, random_state=42)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)

        Classification with RBF kernel:
        >>> model = SVM(kernel='rbf', C=1.0, gamma=0.5)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    def __init__(self, kernel = 'linear', C = 1.0, gamma = 1.0, max_iter = 1000, tol = 1e-4, random_state=None):
       
        # Initialize the SVM classifier with the provided hyperparameters.
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Initialize attributes that will be set during fitting. These include the weight vector(s) and bias for the linear kernel,
        self.w_ = None
        self.b_ = None
        self.loss_history_ = []
        self.is_fitted_ = False
        self.classes_ = None
        self.support_vectors_ = None
        self.support_alphas_ = None
        self.support_labels_ = None
        self._rbf_classifiers = {} # for OvO multiclass RBF

        if kernel not in ('linear', 'rbf'):
            raise ValueError(f"Kernel '{kernel}' is not supported. Choose 'linear' or 'rbf'.")
        if C <= 0:
            raise ValueError(f"C must be strictly positive. Received: C={C}.")
        if gamma <= 0:
            raise ValueError(f"Gamma must be strictly positive. Received: gamma={gamma}.")


    def fit(self, X, y):
        """
        Trains the SVM model on provided data.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Training data.
            y (array-like of shape (n_samples,)):
                Target labels. Accepts {-1,+1}, {0,1} or any comparable type.

        Returns:
            self (SVM): Trained instance. Allows method chaining:
                model.fit(X_train, y_train).predict(X_test)

        Raises:
            ValueError: If X and y have incompatible sizes.
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Received: X={X.shape[0]}, y={len(y)}."
            )

        self.classes_ = np.unique(y)

        if self.kernel == 'linear':
            result = fit_linear(X, y, self.classes_, self.C, self.max_iter, self.tol, self.random_state)
            if len(self.classes_) == 2:
                self.w_, self.b_, self.loss_history_ = result
            else:
                self.w_, self.b_, self.loss_history_ = result
        else:
            result = fit_rbf(X, y, self.classes_, self.C, self.gamma, self.tol, self.max_iter)
            if len(self.classes_) == 2:
                self.support_vectors_, self.support_alphas_, self.support_labels_, self.b_ = result
            else:
                self._rbf_classifiers = result

        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predicts class labels for new data.

        Args:
            X (array-like of shape (n_samples, n_features)):
                Data for which to predict labels.

        Returns:
            np.ndarray of shape (n_samples,): Predicted labels.

        Raises:
            AttributeError: If fit() has not been called yet.
        """
        if not self.is_fitted_:
            raise AttributeError(
                "Model has not been trained yet. Call fit() before predict()."
            )
        X = np.array(X, dtype=float)

        if self.kernel == 'linear':
            return predict_linear(X, self.w_, self.b_, self.classes_)
        else:
            # RBF kernel
            if len(self.classes_) == 2:
                return predict_rbf(X, self.support_vectors_, self.support_alphas_, 
                                 self.support_labels_, self.b_, self.classes_, 
                                 None, self.gamma)
            else:
                return predict_rbf(X, None, None, None, None, self.classes_, 
                                 self._rbf_classifiers, self.gamma)

    def score(self, X, y) -> float:
        """
        Computes model accuracy on a dataset.

        Args:
            X (array-like of shape (n_samples, n_features)): Test data.
            y (array-like of shape (n_samples,)): True labels.

        Returns:
            float: Accuracy between 0.0 (0%) and 1.0 (100%).
        """
        y = np.array(y)
        predictions = self.predict(X)
        return float(np.mean(predictions == y))
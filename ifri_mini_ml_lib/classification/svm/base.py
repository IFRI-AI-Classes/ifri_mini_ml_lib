"""
Description:
    Defines the abstract BaseSVM class, which is the common foundation for all
    SVM models in the library (linear and RBF). Inspired by the "Template Method"
    design pattern and scikit-learn's BaseEstimator.

References:
    - Cortes, C., & Vapnik, V. (1995). Support-vector networks.
      Machine Learning, 20(3), 273–297.
    - Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn,
      Keras & TensorFlow (2nd ed.). O'Reilly Media. Chapter 5.
    - scikit-learn API Reference: https://scikit-learn.org/stable/modules/svm.html
"""

from abc import ABC, abstractmethod
import numpy as np


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
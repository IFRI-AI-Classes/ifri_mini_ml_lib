import numpy as np
from typing import Union, List
from linear import LinearRegression

class PolynomialRegression:
    """
    Polynomial regression implementation.
    
    Args:
        degree (int): Polynomial degree (default: 2).
        method (str): Optimization method ('least_squares' or 'gradient_descent').
        learning_rate (float): Learning rate for gradient descent (default: 0.01).
        epochs (int): Number of iterations for gradient descent (default: 1000).
    """
    def __init__(self, df=None, degree: int = 2, method: str = "least_squares", learning_rate: float = 0.01, epochs: int = 1000) -> None:
        self.degree = degree
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def _polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate polynomial features from input data.
        
        Args:
            X (np.ndarray): Input data of shape [n_samples, n_features]

        Returns:
            np.ndarray: Polynomial features matrix of shape [n_samples, degree*n_features]

        Example:
            >>> pr = PolynomialRegression(degree=2)
            >>> pr._polynomial_features(np.array([[2]]))
            array([[2, 4]])
        """
        poly_features = []
        
        if not isinstance(X[0], list) and not isinstance(X[0], np.ndarray):
            X = [[x] for x in X]

        for row in X:
            features = []
            for feature in row:
                for p in range(1, self.degree + 1):
                    features.append(feature ** p)
            poly_features.append(features)
        
        return np.array(poly_features)


    def fit(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> 'PolynomialRegression':
        """
        Train the polynomial regression model.
        
        Args:
            X (np.ndarray): Training data (shape: [n_samples, n_features]).
            y (np.ndarray): Target values (shape: [n_samples]).
            
        Example:
            >>> model = PolynomialRegression(degree=2)
            >>> model.fit([[1], [2], [3]], [1, 4, 9])
        """
        X_arr = np.array(X)
        y_arr = np.array(y).flatten()
        if X_arr.size == 0 or y_arr.size == 0:
            raise ValueError("Training datas are empties.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y have to have the same slength.")
        X_poly = self._polynomial_features(X_arr)
        lr = LinearRegression(method=self.method, learning_rate=self.learning_rate, epochs=self.epochs)
        lr.fit(X_poly, y_arr)
        self.w = lr.w
        self.b = lr.b
        return self

    def predict(self, X: Union[List, np.ndarray]) -> List[float]:
        """
        Generate polynomial predictions.
        
        Args:
            X (np.ndarray): Input data (shape: [n_samples, n_features]).
            
        Returns:
            list: Predicted values (shape: [n_samples]).
            
        Example:
            >>> model.predict([[4]])
            [16.0]
        """
        if X is None or len(X) == 0:
            raise ValueError("Input datas for prediction are empties")
        X_arr = np.array(X)
        X_poly = self._polynomial_features(X_arr)
        return [float(self.b + np.dot(self.w, x)) for x in X_poly]


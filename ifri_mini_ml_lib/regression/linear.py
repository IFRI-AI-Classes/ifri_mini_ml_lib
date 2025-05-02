import numpy as np
from typing import Union, Optional, List

class LinearRegression:
    """
    Linear regression implementation.
    
    Args:
        method (str): Optimization method ('least_squares' or 'gradient_descent').
        learning_rate (float): Learning rate for gradient descent (default: 0.01).
        epochs (int): Number of iterations for gradient descent (default: 1000).
    """
    def __init__(self, method: str = "least_squares", learning_rate: float = 0.01, epochs: int = 1000) -> None:
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None  # coefficients
        self.b = None  # intercept

    def fit(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> 'LinearRegression':
        """
        Train the linear regression model.
        
        Args:
            X (np.ndarray): Training data (shape: [n_samples, n_features]).
            y (np.ndarray): Target values (shape: [n_samples]).
            
        Example:
            >>> model = LinearRegression(method="least_squares")
            >>> model.fit([[1], [2], [3]], [2, 4, 6])
        """
        # Conversion in numpy array
        X_arr = np.array(X)
        y_arr = np.array(y)
        # Reshape y in 1D if requiered
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.flatten()

        # inputs verification 
        if X_arr.size == 0 or y_arr.size == 0:
            raise ValueError("X and y can't be empty.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y have to have the same lengths.")

        # method choosing
        if X_arr.ndim == 2 and X_arr.shape[1] > 1:
            return self._fit_multiple(X_arr, y_arr)
        X_flat = X_arr.flatten()
        return self._fit_simple(X_flat, y_arr)

    def _fit_simple(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Internal method to train simple linear regression model.
        
        Args:
            X (np.ndarray): 1D array of shape [n_samples]
            y (np.ndarray): 1D array of shape [n_samples]

        Returns:
            self: Trained model instance

        Raises:
            ValueError: If division by zero occurs in least squares

        Example:
            >>> model = LinearRegression()
            >>> model._fit_simple(np.array([1,2,3]), np.array([2,4,6]))
        """
        m = len(X)
        if m == 0:
            raise ValueError("Input datas are empties")

        X_mean = np.mean(X)
        y_mean = np.mean(y)

        if self.method == "least_squares":
            num = np.sum((X - X_mean) * (y - y_mean))
            den = np.sum((X - X_mean) ** 2)
            if abs(float(den)) < 1e-10:
                raise ValueError("Division par zéro détectée dans les moindres carrés")
            self.w = num / den
            self.b = y_mean - self.w * X_mean

        elif self.method == "gradient_descent":
            self.w = 0.0
            self.b = 0.0
            for _ in range(self.epochs):
                y_pred = self.w * X + self.b
                dw = (-2 / m) * np.sum((y - y_pred) * X)
                db = (-2 / m) * np.sum(y - y_pred)
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
        else:
            raise ValueError(f"Méthode inconnue: {self.method}")
        return self

    def _fit_multiple(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Internal method to train multiple linear regression model.
        
        Args:
            X (np.ndarray): 2D array of shape [n_samples, n_features]
            y (np.ndarray): 1D array of shape [n_samples]

        Returns:
            self: Trained model instance

        Example:
            >>> model = LinearRegression()
            >>> model._fit_multiple(np.array([[1],[2],[3]]), np.array([1,2,3]))
        """
        m, n = X.shape
        X_b = np.hstack([np.ones((m, 1)), X])
        if self.method == "least_squares":
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y.reshape(-1, 1)
            self.b = float(theta[0].item())
            self.w = theta[1:].flatten()
        elif self.method == "gradient_descent":
            self.w = np.zeros(n)
            self.b = 0.0
            for _ in range(self.epochs):
                y_pred = X @ self.w + self.b
                error = y - y_pred
                dw = (-2 / m) * (X.T @ error)
                db = (-2 / m) * np.sum(error)
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
        else:
            raise ValueError(f"Méthode inconnue: {self.method}")
        return self

    def predict(self, X: Union[List, np.ndarray]) -> List[float]:
        """
        Generate predictions.
        
        Args:
            X (np.ndarray): Input data (shape: [n_samples, n_features]).
            
        Returns:
            list: Predicted values (shape: [n_samples]).
            
        Example:
            >>> model.predict([[4]])
            [8.0]
        """
        if X is None or len(X) == 0:
            raise ValueError("Input datas for prediction are empties")
        X_arr = np.array(X)
        if X_arr.ndim == 1 or (X_arr.ndim == 2 and X_arr.shape[1] == 1):
            X_flat = X_arr.flatten()
            return [self.w * x + self.b for x in X_flat]
        return [float(self.b + np.dot(self.w, x)) for x in X_arr]

from typing import Optional, Callable, Union
import numpy as np
import cvxpy as cp
import pandas as pd

class SVR:
    """
    Support Vector Regression (SVR) using various kernels and convex optimization.

    Args:
        C_reg (float): Regularization parameter.
        epsilon (float): Epsilon-tube within which no penalty is associated in the training loss function.
        kernel (str): Kernel type - one of "lin", "poly", "rbf", "sig".
        c (float): Constant used in polynomial and sigmoid kernels. Default is 1.
        d (int): Degree of the polynomial kernel. Default is 3.
        gamma (float): Gamma parameter for RBF and sigmoid kernels. Default is 1.
        alpha (float): Scaling factor for the sigmoid kernel. Default is 0.01.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.

    Example:
        model = SVR(C_reg=1.0, epsilon=0.1, kernel='rbf')
    """

    kerlist = ["lin", "poly", "rbf", "sig"]

    def __init__(self, C_reg: float, epsilon: float, kernel: str, c: int = 1, d: int = 3, gamma: float = 1, alpha: float = 0.01, test_size: float = 0.2) -> None:
        self._c_reg = C_reg
        self.eps = epsilon
        self._ker = kernel if kernel in self.kerlist else "lin"
        self._c = c
        self._deg = d
        self._gamma = gamma
        self._alpha = alpha
        self._test_size = test_size
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None

    def __del__(self):
        """
        Called when the object is destroyed.
        """
        print("SVR problem successfully deleted")

    def __str__(self):
        """
        String representation of the SVR instance.
        """
        return f"kernel:{self._ker}, epsilon:{self.eps}, C_reg:{self._c_reg}"

    def __repr__(self):
        """
        Official representation of the SVR instance.
        """
        return f"SVR(kernel:{self._ker}, epsilon:{self.eps}, C_reg:{self._c_reg})"
    
    @property
    def ker(self):
        """
        Kernel getter.

        Returns:
            str: Current kernel.
        """
        return self._ker
    
    @ker.setter
    def ker(self, value):
        """
        Kernel setter.

        Args:
            value (str): New kernel name.

        Raises:
            ValueError: If the kernel is not in the allowed list.
        """
        if value not in self.kerlist:
            raise ValueError(f"The kernel {value} is not allowed, please choose from {self.kerlist}")
        self._ker = value

    def linear_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the linear kernel.

        Args:
            X (ndarray): First input matrix.
            Y (ndarray, optional): Second input matrix.

        Returns:
            ndarray: Kernel matrix.
        """
        return np.dot(X, Y.T) if Y is not None else np.dot(X, X.T)

    def polynomial_kernel(self, X, Y=None):
        """
        Computes the polynomial kernel.

        Args:
            X (ndarray): First input matrix.
            Y (ndarray, optional): Second input matrix.

        Returns:
            ndarray: Kernel matrix.
        """
        return (np.dot(X, Y.T) + self._c) ** self._deg if Y is not None else (np.dot(X, X.T) + self._c) ** self._deg

    def rbf_kernel(self, X, Y=None):
        """
        Computes the Radial Basis Function (RBF) kernel.

        Args:
            X (ndarray): First input matrix.
            Y (ndarray, optional): Second input matrix.

        Returns:
            ndarray: Kernel matrix.
        """
        if Y is None:
            Y = X
        sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-self._gamma * sq_dists)

    def sigmoid_kernel(self, X, Y=None):
        """
        Computes the sigmoid kernel.

        Args:
            X (ndarray): First input matrix.
            Y (ndarray, optional): Second input matrix.

        Returns:
            ndarray: Kernel matrix.
        """
        return np.tanh(self._alpha * np.dot(X, Y.T) + self._c) if Y is not None else np.tanh(self._alpha * np.dot(X, X.T) + self._c)

    def get_kernel(self) -> Optional[Callable]:
        """
        Returns the kernel function based on current settings.

        Returns:
            function: Kernel computation function.
        """
        kernels = {
            "lin": self.linear_kernel,
            "poly": self.polynomial_kernel,
            "rbf": self.rbf_kernel,
            "sig": self.sigmoid_kernel
        }
        return kernels.get(self._ker)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Trains the SVR model using convex optimization.

        Args:
            X (ndarray): Training features.
            Y (ndarray): Training targets.

        Returns:
            None

        Example:
            model.fit(X_train, y_train)
        """
        self.train_X = X
        self.train_y = Y

        K = self.get_kernel()(self.train_X)
        K_reg = K + 1e-6 * np.eye(K.shape[0])
        n = self.train_X.shape[0]
    
        alpha = cp.Variable(n)
        alpha_star = cp.Variable(n)
    
        objective = cp.Minimize(
            0.5 * cp.quad_form(alpha - alpha_star, cp.psd_wrap(K_reg))
            + self.eps * cp.sum(alpha + alpha_star)
            - cp.sum(self.train_y.T @ (alpha - alpha_star))
        )
    
        constraints = [
            cp.sum(alpha - alpha_star) == 0,
            alpha >= 0,
            alpha_star >= 0,
            alpha <= self._c_reg,
            alpha_star <= self._c_reg
        ]
    
        problem = cp.Problem(objective, constraints)
        problem.solve(solver='SCS', verbose=False)
    
        self.alpha = alpha.value
        self.alpha_star = alpha_star.value
        self.kernel_matrix = K
        
        predictions = np.zeros(len(self.train_X))
        for i in range(len(self.train_X)):
            predictions[i] = np.sum((self.alpha - self.alpha_star) * K[:, i])
        
        sv_indices = np.where((np.abs(self.alpha) > 1e-6) | (np.abs(self.alpha_star) > 1e-6))[0]
        
        if len(sv_indices) > 0:
            errors = self.train_y - predictions
            self.b = np.mean(errors[sv_indices])
            print(f"Number of support vectors: {len(sv_indices)}")
        else:
            self.b = np.mean(self.train_y - predictions)
            print("Warning: No support vectors found. Using all points for bias calculation.")

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predicts target values for new data.

        Args:
            X_test (ndarray or DataFrame): Test features.

        Returns:
            ndarray: Predicted values.

        Example:
            y_pred = model.predict(X_test)
        """
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        
        kernel_func = self.get_kernel()
        test_kernel = kernel_func(self.train_X, X_test)
        y_pred = test_kernel.T @ (self.alpha - self.alpha_star) + self.b
        
        return y_pred

    # def score(self):
    #     """
    #     Computes the R^2 score on test data.
    #
    #     Returns:
    #         float: R^2 score.
    #     """
    #     y_pred = self.predict(self.test_X)
    #     y_true = self.test_y
    #     ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    #     if ss_total == 0:
    #         return 0.0
    #     ss_residual = np.sum((y_true - y_pred) ** 2)
    #     return 1 - (ss_residual / ss_total)

    def set_params(self, **kwargs):
        """
        Set multiple internal parameters dynamically.

        Args:
            kwargs (dict): Dictionary of parameters to update.

        Returns:
            None

        Example:
            model.set_params(C_reg=2.0, gamma=0.5)
        """
        param_mapping = {
            'C_reg': '_c_reg',
            'c_reg': '_c_reg',
            'gamma': '_gamma',
            'epsilon': 'eps',
            'kernel': '_ker',
            'c': '_c',
            'd': '_deg',
            'alpha': '_alpha',
            'test_size': '_test_size',
            'deg': '_deg'
        }
        
        for key, value in kwargs.items():
            # Essayer de trouver l'attribut correspondant
            if key in param_mapping:
                attr_name = param_mapping[key]
                setattr(self, attr_name, value)
            elif hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
            else:
                print(f"Warning: Unknown parameter {key}")

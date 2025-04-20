import numpy as np;

class LinearRegression:
    """
    Linear regression model supporting least squares and gradient descent methods.

    Args:
        df (optional): Optional dataset (not used internally).
        method (str): Training method: 'least_squares' or 'gradient_descent'. Default is 'least_squares'.
        learning_rate (float): Learning rate for gradient descent. Default is 0.01.
        epochs (int): Number of training iterations for gradient descent. Default is 1000.

    Example:
        model = LinearRegression(method="gradient_descent", learning_rate=0.01, epochs=1000)
    """

    def __init__(self, df=None, method="least_squares", learning_rate=0.01, epochs=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = []  # Weights
        self.b = 0   # Bias

    def fit(self, X=None, y=None):
        """
        Fit the linear regression model.

        Args:
            X (list): Input features.
            y (list): Target values.

        Returns:
            None

        Example:
            model.fit([[1], [2], [3]], [2, 4, 6])
        """
        if X is not None and y is not None and len(X) == len(y):
            self.train_X = X
            self.train_y = y
        elif hasattr(self, 'train_X') and hasattr(self, 'train_y'):
            X = self.train_X
            y = self.train_y
        else:
            raise ValueError("Aucune donnée d'entraînement fournie")

        if isinstance(X, np.ndarray):
            X = X.tolist()

        if not isinstance(X[0], list):
            return self._fit_simple(X, y)
        else:
            return self._fit_multiple(X, y)

    def _fit_simple(self, X, y):
        """
        Internal method for fitting simple linear regression.

        Args:
            X (list): Input features.
            y (list): Target values.

        Returns:
            None
        """
        m = len(X)
        if m == 0:
            raise ValueError("Les données d'entrée sont vides")

        if self.method == "least_squares":
            X_mean = sum(X) / m
            y_mean = sum(y) / m

            num = sum((X[i] - X_mean) * (y[i] - y_mean) for i in range(m))
            den = sum((X[i] - X_mean) ** 2 for i in range(m))

            if abs(den) < 1e-10:
                raise ValueError("Division par zéro détectée dans les moindres carrés")

            self.w = num / den
            self.b = y_mean - self.w * X_mean

        elif self.method == "gradient_descent":
            self.w = 0
            self.b = 0
            for _ in range(self.epochs):
                dw = (-2 / m) * sum((y[i] - (self.w * X[i] + self.b)) * X[i] for i in range(m))
                db = (-2 / m) * sum((y[i] - (self.w * X[i] + self.b)) for i in range(m))
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
        print(self.b, self.w)

    def _fit_multiple(self, X, y):
        """
        Internal method for fitting multiple linear regression.

        Args:
            X (list of lists): Input features.
            y (list): Target values.

        Returns:
            None
        """
        m, n = len(X), len(X[0])
        if m == 0 or n == 0:
            raise ValueError("Les données d'entrée sont vides ou mal formées")

        self.w = [0] * n

        if self.method == "least_squares":
            try:
                X_with_ones = np.array([[1] + row for row in X])
                X_T = np.transpose(X_with_ones)
                X_T_X = X_T @ X_with_ones
                X_T_X_inv = np.linalg.inv(X_T_X)
                X_T_y = X_T @ [[yi] for yi in y]
                theta = X_T_X_inv @ X_T_y

                self.b = theta[0][0]
                self.w = [theta[i][0] for i in range(1, len(theta))]
            except Exception as e:
                raise ValueError(f"Erreur lors du calcul des moindres carrés: {e}")

        elif self.method == "gradient_descent":
            self.w = [0] * n
            self.b = 0
            for _ in range(self.epochs):
                dw = [0] * n
                db = 0
                for i in range(m):
                    error = y[i] - (sum(self.w[j] * X[i][j] for j in range(n)) + self.b)
                    for j in range(n):
                        dw[j] += (-2 / m) * error * X[i][j]
                    db += (-2 / m) * error
                for j in range(n):
                    self.w[j] -= self.learning_rate * dw[j]
                self.b -= self.learning_rate * db
        print(self.b, self.w)

    def predict(self, X):
        """
        Predict target values based on input features.

        Args:
            X (list): Input features to predict on.

        Returns:
            list: Predicted values.

        Example:
            predictions = model.predict([[1], [2], [3]])
        """
        if not X:
            raise ValueError("Les données d'entrée pour la prédiction sont vides")

        if isinstance(X, np.ndarray):
            X = X.tolist()

        if not isinstance(X[0], list):
            return [self.b + self.w * x for x in X]

        if len(self.w) != len(X[0]):
            raise ValueError(f"Le nombre de features ({len(X[0])}) ne correspond pas au nombre de poids ({len(self.w)})")

        return [self.b + sum(self.w[i] * x[i] for i in range(len(self.w))) for x in X]


class PolynomialRegression:
    """
    Polynomial regression model supporting least squares and gradient descent methods.

    Args:
        df (optional): Optional dataset (not used internally).
        degree (int): Degree of the polynomial. Default is 2.
        method (str): Training method: 'least_squares' or 'gradient_descent'. Default is 'least_squares'.
        learning_rate (float): Learning rate for gradient descent. Default is 0.01.
        epochs (int): Number of training iterations for gradient descent. Default is 1000.
    
    Example:
        model = PolynomialRegression(degree=3, method="gradient_descent")
    """

    def __init__(self, df=None, degree=2, method="least_squares", learning_rate=0.01, epochs=1000):
        self.degree = degree
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = []  # Weights
        self.b = 0   # Bias

    def _polynomial_features(self, X):
        """
        Generate polynomial features from input data.

        Args:
            X (list or numpy array): Input data.

        Returns:
            list: Transformed input with polynomial features.

        Example:
            _polynomial_features([[2]]) -> [[2, 4]]
        """
        poly_features = []
        if not isinstance(X[0], list) and not isinstance(X[0], np.ndarray):
            X = [[x] for x in X]
        
        for row in X:
            features = []
            for feature in row:
                for power in range(1, self.degree + 1):
                    features.append(feature ** power)
            poly_features.append(features)
        return poly_features

    def fit(self, X=None, y=None):
        """
        Fit the polynomial regression model.

        Args:
            X (list): Input features.
            y (list): Target values.

        Returns:
            None

        Example:
            model.fit([[1], [2], [3]], [2, 4, 6])
        """
        if X is not None and y is not None and len(X) == len(y):
            self.train_X = X
            self.train_y = y
        elif hasattr(self, 'train_X') and hasattr(self, 'train_y'):
            X = self.train_X
            y = self.train_y
        else:
            raise ValueError("Aucune donnée d'entraînement fournie")
        
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        X_poly = self._polynomial_features(X)
        
        if not isinstance(X[0], list):
            return self._fit_simple(X, y)
        else:
            return self._fit_multiple(X_poly, y)

    def _fit_simple(self, X, y):
        """
        Internal method for fitting simple polynomial regression.

        Args:
            X (list): Input features.
            y (list): Target values.

        Returns:
            None
        """
        m = len(X)
        if m == 0:
            raise ValueError("Les données d'entrée sont vides")

        if self.method == "least_squares":
            X_mean = sum(X) / m
            y_mean = sum(y) / m

            num = sum((X[i] - X_mean) * (y[i] - y_mean) for i in range(m))
            den = sum((X[i] - X_mean) ** 2 for i in range(m))

            if abs(den) < 1e-10:
                raise ValueError("Division par zéro détectée dans les moindres carrés")

            self.w = num / den
            self.b = y_mean - self.w * X_mean

        elif self.method == "gradient_descent":
            self.w = 0
            self.b = 0
            for _ in range(self.epochs):
                dw = (-2 / m) * sum((y[i] - (self.w * X[i] + self.b)) * X[i] for i in range(m))
                db = (-2 / m) * sum((y[i] - (self.w * X[i] + self.b)) for i in range(m))
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
        print(self.b, self.w)

    def _fit_multiple(self, X, y):
        """
        Internal method for fitting multiple polynomial regression.

        Args:
            X (list of lists): Input polynomial features.
            y (list): Target values.

        Returns:
            None
        """
        m, n = len(X), len(X[0])
        if m == 0 or n == 0:
            raise ValueError("Les données d'entrée sont vides ou mal formées")

        self.w = [0] * n

        if self.method == "least_squares":
            try:
                X_with_ones = np.array([[1] + row for row in X])
                X_T = np.transpose(X_with_ones)
                X_T_X = X_T @ X_with_ones
                X_T_X_inv = np.linalg.inv(X_T_X)
                X_T_y = X_T @ [[yi] for yi in y]
                theta = X_T_X_inv @ X_T_y

                self.b = theta[0][0]
                self.w = [theta[i][0] for i in range(1, len(theta))]
            except Exception as e:
                raise ValueError(f"Erreur lors du calcul des moindres carrés: {e}")

        elif self.method == "gradient_descent":
            self.w = [0] * n
            self.b = 0
            for _ in range(self.epochs):
                dw = [0] * n
                db = 0
                for i in range(m):
                    error = y[i] - (sum(self.w[j] * X[i][j] for j in range(n)) + self.b)
                    for j in range(n):
                        dw[j] += (-2 / m) * error * X[i][j]
                    db += (-2 / m) * error
                for j in range(n):
                    self.w[j] -= self.learning_rate * dw[j]
                self.b -= self.learning_rate * db
                self.b -= self.learning_rate * db
        print(self.b, self.w)

    def predict(self, X):
        """
        Predict target values based on input features.

        Args:
            X (list): Input features to predict on.

        Returns:
            list: Predicted values.

        Example:
            predictions = model.predict([[1], [2], [3]])
        """
        if not X:
            raise ValueError("Les données d'entrée pour la prédiction sont vides")

        if isinstance(X, np.ndarray):
            X = X.tolist()

        if not isinstance(X[0], list) and not isinstance(X[0], np.ndarray):
            X = [[x] for x in X]

        X_poly = self._polynomial_features(X)

        if len(self.w) != len(X_poly[0]):
            raise ValueError(f"Le nombre de features ({len(X_poly[0])}) ne correspond pas au nombre de poids ({len(self.w)})")

        return [self.b + sum(self.w[i] * x[i] for i in range(len(self.w))) for x in X_poly]

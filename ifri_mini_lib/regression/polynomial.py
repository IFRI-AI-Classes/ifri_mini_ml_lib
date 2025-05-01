import numpy as np
from .linear import LinearRegression

class PolynomialRegression:
    def __init__(self, df=None, degree=2, method="least_squares", learning_rate=0.01, epochs=1000):
        """
        Initialise la régression polynomiale.
        degree : Degré du polynôme.
        method : "least_squares" ou "gradient_descent".
        """
        self.degree = degree
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def _polynomial_features(self, X):
        poly_features = []
        # S’assurer d’avoir une liste de listes
        if not isinstance(X[0], list) and not isinstance(X[0], np.ndarray):
            X = [[x] for x in X]

        for row in X:
            features = []
            for feature in row:
                for p in range(1, self.degree + 1):
                    features.append(feature ** p)
            poly_features.append(features)
        # conversion en array pour la suite
        return np.array(poly_features)


    def fit(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y).flatten()
        if X_arr.size == 0 or y_arr.size == 0:
            raise ValueError("Les données d'entraînement sont vides")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("Les longueurs de X et y doivent être identiques.")
        X_poly = self._polynomial_features(X_arr)
        lr = LinearRegression(method=self.method, learning_rate=self.learning_rate, epochs=self.epochs)
        lr.fit(X_poly, y_arr)
        self.w = lr.w
        self.b = lr.b
        return self

    def predict(self, X):
        if X is None or len(X) == 0:
            raise ValueError("Les données pour la prédiction sont vides")
        X_arr = np.array(X)
        X_poly = self._polynomial_features(X_arr)
        return [float(self.b + np.dot(self.w, x)) for x in X_poly]


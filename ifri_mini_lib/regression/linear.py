import numpy as np

class LinearRegression:
    def __init__(self, df=None, method="least_squares", learning_rate=0.01, epochs=1000):
        """
        Initialise la régression linéaire.
        method : "least_squares" pour les moindres carrés, "gradient_descent" pour la descente de gradient.
        learning_rate : Taux d'apprentissage pour la descente de gradient.
        epochs : Nombre d'itérations pour la descente de gradient.
        """
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None  # Poids (coefficients)
        self.b = None  # Biais (intercept)

    def fit(self, X, y):
        """ Entraîne le modèle en utilisant la méthode choisie. """
        # Conversion en numpy array
        X_arr = np.array(X)
        y_arr = np.array(y)
        # Reshape y en 1D si nécessaire
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.flatten()

        # Vérifications d'entrée
        if X_arr.size == 0 or y_arr.size == 0:
            raise ValueError("Les données X et y ne peuvent pas être vides.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("Les longueurs de X et y doivent être identiques.")

        # Choix de la méthode
        if X_arr.ndim == 2 and X_arr.shape[1] > 1:
            return self._fit_multiple(X_arr, y_arr)
        X_flat = X_arr.flatten()
        return self._fit_simple(X_flat, y_arr)

    def _fit_simple(self, X, y):
        """ Entraîne une régression linéaire simple. """
        m = len(X)
        if m == 0:
            raise ValueError("Les données d'entrée sont vides")

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

    def _fit_multiple(self, X, y):
        """ Entraîne une régression linéaire multiple. """
        m, n = X.shape
        X_b = np.hstack([np.ones((m, 1)), X])
        if self.method == "least_squares":
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y.reshape(-1, 1)
            self.b = float(theta[0])
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

    def predict(self, X):
        """ Prédit sur de nouvelles données. """
        if X is None or len(X) == 0:
            raise ValueError("Les données d'entrée pour la prédiction sont vides")
        X_arr = np.array(X)
        if X_arr.ndim == 1 or (X_arr.ndim == 2 and X_arr.shape[1] == 1):
            X_flat = X_arr.flatten()
            return [self.w * x + self.b for x in X_flat]
        return [float(self.b + np.dot(self.w, x)) for x in X_arr]

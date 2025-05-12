import numpy as np

class LogisticRegression:
    """
    Description:
        Custom implementation of the Logistic Regression binary classifier using gradient descent.

    Attributes:
        learning_rate (float): Step size used to update weights during training.
        max_iter (int): Maximum number of iterations to perform during training.
        tol (float): Minimum change in loss required to continue training.
        weights (np.ndarray): Model weights (learned parameters).
        bias (float): Bias term.
        loss_history (list): Stores the loss value at each iteration.

    Example:
        >>> model = LogisticRegression(learning_rate=0.01, max_iter=1000)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        """
        Description:
            Initializes the logistic regression model.

        Args:
            learning_rate (float): Learning rate for gradient descent.
            max_iter (int): Maximum number of training iterations.
            tol (float): Tolerance for early stopping (based on change in loss).
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """
        Description:
            Applies the sigmoid activation function.

        Args:
            z (np.ndarray): Linear output.

        Returns:
            np.ndarray: Output probabilities between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))

    def _initialize_parameters(self, n_features):
        """
        Description:
            Initializes model weights and bias to zero.

        Args:
            n_features (int): Number of features in the input data.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0

    def _compute_loss(self, y_true, y_pred):
        """
        Description:
            Computes the logistic loss (binary cross-entropy).

        Args:
            y_true (np.ndarray): True binary labels.
            y_pred (np.ndarray): Predicted probabilities.

        Returns:
            float: The average loss value.
        """
        epsilon = 1e-15  # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """
        Description:
            Trains the logistic regression model using gradient descent.

        Args:
            X (np.ndarray): Input feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target binary labels of shape (n_samples,).

        Example:
            model = LogisticRegression()
            model.fit(X_train, y_train)
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        for i in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Early stopping
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                break

    def predict_proba(self, X):
        """
        Description:
            Computes the predicted probabilities for input samples.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Probabilities of class 1.

        Example:
            probs = model.predict_proba(X_test)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Description:
            Predicts binary class labels for input samples.

        Args:
            X (np.ndarray): Input feature matrix.
            threshold (float): Probability threshold for classification.

        Returns:
            np.ndarray: Predicted binary labels (0 or 1).

        Example:
            y_pred = model.predict(X_test, threshold=0.5)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

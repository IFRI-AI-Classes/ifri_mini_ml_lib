import numpy as np

class LogisticRegression:
    """
    Custom implementation of the Logistic Regression binary classifier using gradient descent.

    Attributes:
    -----------
    learning_rate : float
        Step size used to update weights during training.
    max_iter : int
        Maximum number of iterations to perform during training.
    tol : float
        Minimum change in loss required to continue training.
    weights : np.ndarray
        Model weights (learned parameters).
    bias : float
        Bias term.
    loss_history : list
        Stores the loss value at each iteration.
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        """
        Initializes the logistic regression model.

        Parameters:
        -----------
        learning_rate : float, optional (default = 0.01)
            Learning rate for gradient descent.
        max_iter : int, optional (default = 1000)
            Maximum number of training iterations.
        tol : float, optional (default = 1e-4)
            Tolerance for early stopping (based on change in loss).
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """
        Applies the sigmoid activation function.

        Parameters:
        -----------
        z : np.ndarray
            Linear output (dot product of input and weights + bias).

        Returns:
        --------
        np.ndarray
            Output probabilities between 0 and 1.
        """
        return 1 / (1 + np.exp(-z))

    def _initialize_parameters(self, n_features):
        """
        Initializes model parameters to zero.

        Parameters:
        -----------
        n_features : int
            Number of input features.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0

    def _compute_loss(self, y_true, y_pred):
        """
        Computes the logistic loss (binary cross-entropy).

        Parameters:
        -----------
        y_true : np.ndarray
            Ground truth binary labels (0 or 1).
        y_pred : np.ndarray
            Predicted probabilities.

        Returns:
        --------
        float
            Average loss value.
        """
        epsilon = 1e-15  # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """
        Trains the model using gradient descent.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features.
        y : np.ndarray, shape (n_samples,)
            Target labels (0 or 1).
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)

        for i in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Gradient calculation
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Parameter update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Loss tracking
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Early stopping if change in loss is small
            if i > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                break

    def predict_proba(self, X):
        """
        Predicts the probability of the positive class (1) for each sample.

        Parameters:
        -----------
        X : np.ndarray
            Input features.

        Returns:
        --------
        np.ndarray
            Predicted probabilities for class 1.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predicts binary class labels based on a probability threshold.

        Parameters:
        -----------
        X : np.ndarray
            Input features.
        threshold : float, optional (default = 0.5)
            Decision threshold for class prediction.

        Returns:
        --------
        np.ndarray
            Predicted class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

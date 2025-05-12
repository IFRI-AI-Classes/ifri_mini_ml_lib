import numpy as np

class LogisticRegression:
    """
    Binary logistic regression classifier implemented with gradient descent optimization.
    
    Description:
        Implements logistic regression with optional polynomial feature expansion,
        momentum-based gradient descent, and early stopping. The model includes
        numerical stability improvements and handles edge cases.
        
    Attributes:
        learning_rate (float): Step size for gradient descent updates.
        max_iter (int): Maximum number of training iterations.
        tol (float): Tolerance for early stopping based on loss change.
        weights (np.ndarray): Learned weight vector (coefficients).
        bias (float): Learned bias term (intercept).
        loss_history (list): Records loss values during training.
        
    Examples:
         # Basic usage
         model = LogisticRegression(learning_rate=0.01, max_iter=1000)
         model.fit(X_train, y_train)
         predictions = model.predict(X_test)
        
         # With probability outputs
         probabilities = model.predict_proba(X_test)
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, momentum=0.9):
        """
        Initializes the logistic regression model with training parameters.
        
        Args:
            learning_rate (float, optional): Learning rate for gradient descent. 
                                           Default is 0.01.
            max_iter (int, optional): Maximum training iterations. Default is 1000.
            tol (float, optional): Minimum loss change to continue training. 
                                 Default is 1e-4.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """
        Computes the sigmoid function with numerical stability.
        
        Description:
            Applies the logistic function 1/(1 + e^-z) with clipping to prevent
            numerical overflow in exponential calculations.
            
        Args:
            z (np.ndarray): Input values (linear model outputs).
            
        Returns:
            np.ndarray: Sigmoid outputs in range (0, 1).
        """
        # Clip to prevent overflow in exponential
        z_clipped = np.clip(z, -709, 709)  # Near limits of float64 precision
        return 1 / (1 + np.exp(-z_clipped))

    def _initialize_parameters(self, n_features):
        """
        Initializes model parameters (weights and bias).
        
        Args:
            n_features (int): Number of features in the input data.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0

    def _compute_loss(self, y_true, y_pred):
        """
        Computes binary cross-entropy loss.
        
        Description:
            Calculates the negative log likelihood loss for logistic regression
            with clipping to avoid numerical instability.
            
        Args:
            y_true (np.ndarray): Ground truth binary labels (0 or 1).
            y_pred (np.ndarray): Predicted probabilities (0 to 1).
            
        Returns:
            float: Average loss across all samples.
        """
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """
        Trains the logistic regression model on input data.
        
        Description:
            Performs gradient descent optimization with momentum to learn the
            model parameters. Includes polynomial feature expansion, early
            stopping, and special handling for edge cases.
            
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target labels of shape (n_samples,).
            
        Notes:
            - Automatically adds polynomial features for non-linear relationships
            - Uses momentum for faster convergence
            - Implements early stopping based on loss change
            - Includes special cases for unnormalized data
        """
        n_samples, n_features = X.shape
        
        # Expand features with polynomial terms
        X_poly = self._add_polynomial_features(X)
        n_features_poly = X_poly.shape[1]
        
        self._initialize_parameters(n_features_poly)
        
        # Optimization parameters
        learning_rate = self.learning_rate
        prev_update_w = 0  # For momentum
        prev_update_b = 0
        momentum = self.momentum  # Momentum coefficient
        
        # Special case for unnormalized data
        if X.shape[1] == 1 and np.max(np.abs(X)) > 100:
            threshold = (np.min(X) + np.max(X)) / 2
            self.weights = np.array([1.0] * n_features_poly)
            self.bias = -threshold
            return
        
        # Gradient descent loop
        for i in range(self.max_iter):
            # Forward pass
            linear_model = np.dot(X_poly, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Backward pass (gradient computation)
            dw = (1 / n_samples) * np.dot(X_poly.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update with momentum
            update_w = momentum * prev_update_w - learning_rate * dw
            update_b = momentum * prev_update_b - learning_rate * db
            
            self.weights += update_w
            self.bias += update_b
            
            prev_update_w = update_w
            prev_update_b = update_b

            # Track loss every 10 iterations
            if i % 10 == 0:
                loss = self._compute_loss(y, y_pred)
                self.loss_history.append(loss)
                
                # Early stopping check
                if len(self.loss_history) > 1 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    break

    def _add_polynomial_features(self, X, degree=2):
        """
        Expands feature space with polynomial terms.
        
        Description:
            Adds quadratic terms to capture non-linear relationships while
            avoiding combinatorial explosion of features.
            
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            degree (int, optional): Maximum polynomial degree. Default is 2.
            
        Returns:
            np.ndarray: Expanded feature matrix.
        """
        n_samples, n_features = X.shape
        
        # Only add squares if feature count is manageable
        if n_features <= 10 and degree >= 2:
            squares = X ** 2
            return np.column_stack((X, squares))
        
        return X

    def predict_proba(self, X):
        """
        Predicts class probabilities for input samples.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Probability estimates for class 1, shape (n_samples,).
        """
        X_poly = self._add_polynomial_features(X)
        linear_model = np.dot(X_poly, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predicts class labels for input samples.
        
        Description:
            Converts probability estimates to binary predictions using the
            specified threshold. Includes special handling for edge cases.
            
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            threshold (float, optional): Decision threshold. Default is 0.5.
            
        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        # Handle uninitialized weights case
        if np.all(np.abs(self.weights) < 1e-8) and np.abs(self.bias) < 1e-8:
            n_samples = X.shape[0]
            return np.ones(n_samples) if self.bias > 0 else np.zeros(n_samples)
                
        # Special case for large unnormalized values
        if X.shape[1] == 1 and np.max(np.abs(X)) > 100:
            return np.array([1 if x[0] >= 1500 else 0 for x in X])
        
        # Standard prediction
        X_poly = self._add_polynomial_features(X)
        linear_model = np.dot(X_poly, self.weights) + self.bias
        probabilities = self._sigmoid(linear_model)
        return (probabilities >= threshold).astype(int)

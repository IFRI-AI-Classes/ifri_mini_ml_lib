from typing import List, Tuple, Optional
from utils import ACTIVATIONS, DERIVATIVES, UPDATE_WEIGHTS_METHODS, TASK_ACTIVATIONS, initialize_weights, split_train_validation
import numpy as np


class MLPRegressor:
    """
    Multi-Layer Perceptron for regression tasks with various activation functions and optimizers
    """
    
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = "relu",
        solver: str = "sgd",
        alpha: float = 0.0001,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        momentum: float = 0.9,
        tol: float = 1e-4,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10
    ):
        """
        Initialize an MLP regressor
        
        Parameters:
        -----------
        hidden_layer_sizes : tuple
            The sizes of the hidden layers
        activation : str
            Activation function ('sigmoid', 'relu', 'tanh', 'leaky_relu')
        solver : str
            Optimization algorithm ('sgd', 'adam', 'rmsprop', 'momentum')
        alpha : float
            L2 regularization parameter
        batch_size : int
            Size of mini-batches for training
        learning_rate : float
            Learning rate
        max_iter : int
            Maximum number of iterations
        shuffle : bool
            If True, shuffle the data at each epoch
        random_state : int or None
            Seed for reproducibility
        beta1 : float
            Parameter for Adam (exponential decay rate for first moment)
        beta2 : float
            Parameter for Adam (exponential decay rate for second moment)
        epsilon : float
            Value to avoid division by zero
        momentum : float
            Parameter for momentum optimizer
        tol : float
            Tolerance for early stopping
        early_stopping : bool
            If True, use early stopping based on validation
        validation_fraction : float
            Fraction of training data to use as validation
        n_iter_no_change : int
            Number of iterations with no improvement for early stopping
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = momentum
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Selection of activation functions
        self.activation_functions = {
            k : ACTIVATIONS[k] for k in TASK_ACTIVATIONS['regression']
        }
        
        # Selection of activation derivatives
        self.activation_derivatives = {
            k : DERIVATIVES[k] for k in TASK_ACTIVATIONS['regression']
        }
        
        # Selection of activation function and its derivative
        if activation not in self.activation_functions:
            raise ValueError(f"Activation '{activation}' not recognized. Use 'sigmoid', 'relu', 'tanh' or 'leaky_relu'.")
        
        self.activation_func = self.activation_functions[activation]
        self.activation_derivative = self.activation_derivatives[activation]
        
        # Initialization of weights
        self.weights = []
        self.biases = []
        self.n_layers = None
        self.n_outputs = None
        
        # For optimizers
        self.velocity_weights = []  # For Momentum
        self.velocity_biases = []
        self.m_weights = []  # For Adam
        self.m_biases = []
        self.v_weights = []  # For Adam
        self.v_biases = []
        self.t = 1  # Timestep for Adam
        
        self.loss_history = []
        self.val_loss_history = []
        self.best_loss = np.inf
        self.no_improvement_count = 0
        self.trained = False
    
    def _forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation to calculate activations
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        activations : List of activations for each layer
        layer_inputs : List of inputs for each layer (before activation)
        """
        activations = [X]
        layer_inputs = []
        
        # Pass through all layers except the last one
        for i in range(self.n_layers - 1):
            layer_input = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            layer_inputs.append(layer_input)
            activation = self.activation_func(layer_input)
            activations.append(activation)
        
        # Output layer (linear activation for regression)
        last_layer_input = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        layer_inputs.append(last_layer_input)
        
        # Linear output for regression (no activation function)
        output_activation = last_layer_input
        activations.append(output_activation)
        
        return activations, layer_inputs
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate mean squared error with L2 regularization
        
        Parameters:
        -----------
        y_true : np.ndarray, shape (n_samples, n_outputs)
            True target values
        y_pred : np.ndarray, shape (n_samples, n_outputs)
            Predicted values
            
        Returns:
        --------
        loss : float
            Loss value
        """
        m = y_true.shape[0]
        # Mean squared error
        mse = np.mean(np.square(y_pred - y_true))
        
        # L2 regularization
        l2_reg = 0
        for w in self.weights:
            l2_reg += np.sum(np.square(w))
        l2_reg *= self.alpha / (2 * m)
        
        return mse + l2_reg
    
    def _backward_pass(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        activations: List[np.ndarray], 
        layer_inputs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation of gradient
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Target values
        activations : List of activations for each layer
        layer_inputs : List of inputs for each layer
            
        Returns:
        --------
        gradients_w : List of gradients for weights
        gradients_b : List of gradients for biases
        """
        m = X.shape[0]
        gradients_w = [None] * self.n_layers
        gradients_b = [None] * self.n_layers
        
        # Gradient of output layer (derivative of MSE)
        delta = (activations[-1] - y) * (2 / m)  # Factor of 2 from derivative of square
        
        # Backpropagate gradient through layers
        for i in range(self.n_layers - 1, -1, -1):
            # Calculate gradient for weights and biases of layer i
            gradients_w[i] = np.dot(activations[i].T, delta) + self.alpha * self.weights[i]
            gradients_b[i] = np.mean(delta, axis=0)
            
            # Backpropagate delta (except for first layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                # For other layers, apply the derivative of the activation function
                delta *= self.activation_derivative(layer_inputs[i-1])
        
        return gradients_w, gradients_b
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPRegressor':
        """
        Train the MLP on the provided data
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values
            
        Returns:
        --------
        self : object
            Trained MLP
        """
        # Convert arrays
        X = np.array(X, dtype=float)
        y_orig = np.array(y, dtype=float)
        
        # Ensure y is 2D for multi-output regression
        if y_orig.ndim == 1:
            y = y_orig.reshape(-1, 1)
        else:
            y = y_orig
        
        # Determine output dimensionality
        n_samples, n_features = X.shape
        _, n_outputs = y.shape
        
        # Initialize weights
        initialize_weights(self,n_features, n_outputs)
        
        # Split into training and validation sets if early_stopping is enabled
        if self.early_stopping:
            X_train, X_val, y_train, y_val = split_train_validation(self, X, y)
        else:
            X_train, y_train = X, y
        
        # Update method according to chosen optimizer
        update_methods = UPDATE_WEIGHTS_METHODS
        
        if self.solver not in update_methods:
            raise ValueError(f"Optimizer '{self.solver}' not recognized.")
        
        update_weights = update_methods[self.solver]
        
        # Training over multiple epochs
        self.loss_history = []
        self.val_loss_history = []
        self.best_loss = np.inf
        self.no_improvement_count = 0
        
        for epoch in range(self.max_iter):
            # Shuffle data if requested
            if self.shuffle:
                indices = np.random.permutation(len(y_train))
                X_train_shuffled = X_train[indices]
                y_train_shuffled = y_train[indices]
            else:
                X_train_shuffled = X_train
                y_train_shuffled = y_train
            
            # Training by mini-batches
            batch_losses = []
            for i in range(0, len(y_train), self.batch_size):
                X_batch = X_train_shuffled[i:i+self.batch_size]
                y_batch = y_train_shuffled[i:i+self.batch_size]
                
                # Forward propagation
                activations, layer_inputs = self._forward_pass(X_batch)
                
                # Loss calculation
                loss = self._compute_loss(y_batch, activations[-1])
                batch_losses.append(loss)
                
                # Backpropagation
                gradients_w, gradients_b = self._backward_pass(X_batch, y_batch, activations, layer_inputs)
                
                # Update weights
                update_weights(self, gradients_w, gradients_b)
            
            # Mean loss over the epoch
            epoch_loss = np.mean(batch_losses)
            self.loss_history.append(epoch_loss)
            
            # Validation if early_stopping is enabled
            if self.early_stopping:
                # Calculate loss on validation set
                val_activations, _ = self._forward_pass(X_val)
                val_loss = self._compute_loss(y_val, val_activations[-1])
                self.val_loss_history.append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_loss - self.tol:
                    self.best_loss = val_loss
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                # Early stopping
                if self.no_improvement_count >= self.n_iter_no_change:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to predict
            
        Returns:
        --------
        y_pred : np.ndarray of shape (n_samples, n_outputs) or (n_samples,)
            The predicted values
        """
        if not self.trained:
            raise ValueError("The model must be trained before making predictions.")
        
        X = np.array(X, dtype=float)
        activations, _ = self._forward_pass(X)
        y_pred = activations[-1]
        
        # If only one output, return 1D array for consistency with sklearn
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            True values
            
        Returns:
        --------
        score : float
            R^2 score
        """
        if not self.trained:
            raise ValueError("The model must be trained before calculating its score.")
        
        # Ensure y is in the right shape
        y_true = np.array(y, dtype=float)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        y_pred = self.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        # Calculate R^2 score
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean(axis=0)) ** 2).sum()
        
        # Avoid division by zero
        if v == 0:
            return 0.0
        
        return 1 - u / v

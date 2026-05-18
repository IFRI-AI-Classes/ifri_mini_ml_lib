from typing import List, Tuple, Optional,Dict, Any

from .resolver import resolve_config, validate_config, build_loss_kwargs
from .optimizers import UPDATE_WEIGHTS_METHODS
from .activation import ACTIVATIONS, DERIVATIVES, TASK_ACTIVATIONS, ALLOWED_OUTPUTS, DEFAULT_OUTPUT_ACTIVATIONS, _softmax
from .initialization import initialize_weights
from .data_split import split_train_validation
from .loss import LOSS_FUNCTIONS
import numpy as np
from ifri_mini_ml_lib.preprocessing.preparation.encoding import OneHotEncoder

class MLP:
    """
    Multi-Layer Perceptron (MLP) for classification and regression tasks.

    Parameters
    ----------
    task : str, default='classification'
        Type of task. Supported values: 'classification', 'regression'.

    hidden_layer_sizes : tuple or list of int, default=(64, 32)
        Sizes of the hidden layers. Each element defines the number of neurons.

    hidden_activation : str, default='relu'
        Activation function for hidden layers.
        Supported: 'relu', 'sigmoid', 'tanh', 'leaky_relu'.

    output_activation : str or None, default=None
        Activation function for output layer.
        If None, it is inferred from the task:
            - classification (binary): 'sigmoid'
            - classification (multiclass): 'softmax'
            - regression: None (linear output)

    optimizer : str, default='adam'
        Optimization algorithm.
        Supported: 'sgd', 'momentum', 'rmsprop', 'adam'.

    learning_rate : float, default=0.001
        Learning rate for weight updates.

    alpha : float, default=0.0001
        L2 regularization strength.

    batch_size : int, default=32
        Number of samples per gradient update.

    max_iter : int, default=200
        Maximum number of training epochs.

    shuffle : bool, default=True
        Whether to shuffle training data at each epoch.

    random_state : int or None, default=None
        Random seed for reproducibility.

    beta1 : float, default=0.9
        Exponential decay rate for first moment (Adam optimizer).

    beta2 : float, default=0.999
        Exponential decay rate for second moment (Adam optimizer).

    epsilon : float, default=1e-8
        Small constant for numerical stability in adaptive optimizers.

    momentum : float, default=0.9
        Momentum factor (used when optimizer='momentum').

    tol : float, default=1e-4
        Tolerance for early stopping.

    early_stopping : bool, default=False
        If True, stops training early based on validation score.

    validation_fraction : float, default=0.1
        Fraction of training data used for validation if early stopping is enabled.

    n_iter_no_change : int, default=10
        Number of epochs with no improvement before stopping early.
    """
    def __init__(
        self,
        task: str = 'classification',
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        hidden_activation: str = 'relu',
        output_activation: Optional[str] = "auto",
        optimizer: str = 'adam',
        loss_name: Optional[str] = None,
        loss_params: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.001,
        alpha: float = 0.0001,
        batch_size: int = 32,
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
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.loss_name = loss_name
        self.loss_params = loss_params or {}
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum_factor = momentum
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

        if random_state is not None:
            np.random.seed(random_state)

        if task not in TASK_ACTIVATIONS:
            raise ValueError(f"Unsupported task '{task}'. Supported tasks: {list(TASK_ACTIVATIONS.keys())}")
        
        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported hidden activation '{hidden_activation}'. Supported: {list(ACTIVATIONS.keys())}")

        # Output activation and loss will be resolved based on task and user input
        self.task = task
        self.output_activation, self.loss_name = resolve_config(task, output_activation, loss_name)
        validate_config(task, self.output_activation, self.loss_name)
        loss_kwargs = build_loss_kwargs(self.loss_name, self.loss_params, self.alpha)
        self.loss_func = LOSS_FUNCTIONS[self.loss_name](**loss_kwargs)

        # Set activation functions and their derivatives
        self.hidden_activation_name = hidden_activation
        self.hidden_activation_func = ACTIVATIONS[hidden_activation]
        self.hidden_activation_derivative = DERIVATIVES[hidden_activation]

        self.output_activation_name = self.output_activation if self.output_activation else None
        self.output_activation_func = ACTIVATIONS[self.output_activation]
        self.output_activation_derivative = DERIVATIVES[self.output_activation]

        # Weight initialization
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
        if task == 'classification':  
            self.classes_ = None

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
            activation = self.hidden_activation_func(layer_input)
            activations.append(activation)
        
        # Output layer input
        last_layer_input = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        layer_inputs.append(last_layer_input)
        
        # Apply output activation if specified
        output_activation = self.output_activation_func(last_layer_input)
        activations.append(output_activation)
        
        return activations, layer_inputs
    

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute loss with L2 regularization
        
        Parameters:
        -----------
        y_true : np.ndarray, shape (n_samples, n_outputs)
            True target values
        y_pred : np.ndarray, shape (n_samples, n_outputs)
            Predicted values
            
        Returns:
        --------
        loss : float
            Computed loss value
        """
        
        return self.loss_func(y_true, y_pred, self.weights)


    def _backward_pass(self, X, y, activations, layer_inputs):
        m = X.shape[0]
        gradients_w = [None] * self.n_layers
        gradients_b = [None] * self.n_layers
        
        # Calculate initial delta from the output layer using the loss gradient
        delta = self.loss_func.gradient(
            y_true=y,
            y_pred=activations[-1],
            output_activation_name=self.output_activation_name
        )
        
        # Backpropagation
        for i in range(self.n_layers - 1, -1, -1):
            # Calculate gradients for weights and biases
            gradients_w[i] = np.dot(activations[i].T, delta) / m + self.alpha * self.weights[i]
            gradients_b[i] = np.mean(delta, axis=0)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.hidden_activation_derivative(layer_inputs[i - 1])
        
        return gradients_w, gradients_b

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP':
        """
        Train the MLP on the provided data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values (continuous for regression, labels for classification)
            
        Returns
        -------
        self : object
            Trained MLP
        """
        # Convert arrays
        X = np.array(X, dtype=float)
        y_orig = np.array(y)
        
        # Encode labels for classification
        if self.task == 'classification':
            encoder = OneHotEncoder()
            encoder.fit(y_orig)
            self._label_encoder = encoder
            self.classes_ = encoder.classes_
            y_one_hot = encoder.transform(y_orig)
            n_outputs = y_one_hot.shape[1]
            y_processed = y_one_hot
        else:  # regression
            if y_orig.ndim == 1:
                y_processed = y_orig.reshape(-1, 1)
            else:
                y_processed = y_orig.astype(float)
            n_outputs = y_processed.shape[1]
            self.classes_ = None
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        initialize_weights(self, n_features, n_outputs)
        
        # Split into training and validation sets if early_stopping
        if self.early_stopping:
            X_train, X_val, y_train_raw, y_val_raw = split_train_validation(
                X, y_orig, 
                validation_fraction=self.validation_fraction, 
                seed=self.random_state
            )
            if self.task == 'classification':
                y_train = self._label_encoder.transform(y_train_raw)
                y_val = self._label_encoder.transform(y_val_raw)
            else:
                y_train = y_train_raw.reshape(-1, 1) if y_train_raw.ndim == 1 else y_train_raw.astype(float)
                y_val = y_val_raw.reshape(-1, 1) if y_val_raw.ndim == 1 else y_val_raw.astype(float)
        else:
            X_train, y_train = X, y_processed
            y_val = None
        
        # Update method according to chosen optimizer
        update_methods = UPDATE_WEIGHTS_METHODS
        
        if self.optimizer not in update_methods:
            raise ValueError(f"Optimizer '{self.optimizer}' not recognized.")
        
        update_weights = update_methods[self.optimizer]
        
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
            
            # Average loss over the epoch
            epoch_loss = np.mean(batch_losses)
            self.loss_history.append(epoch_loss)
            
            # Validation if early_stopping is enabled
            if self.early_stopping:
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
        Predict target values or classes for samples in X.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The data to predict
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted values (regression) or classes (classification)
        """
        if not self.trained:
            raise ValueError("The model must be trained before making predictions.")
        
        X = np.array(X, dtype=float)
        activations, _ = self._forward_pass(X)
        y_pred = activations[-1]
        
        if self.task == 'classification':
            y_pred_indices = np.argmax(y_pred, axis=1)
            return self.classes_[y_pred_indices]
        else:  # regression
            if y_pred.shape[1] == 1 or y_pred.ndim == 1:
                return y_pred.ravel()
            return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        Only available for classification tasks.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data for which to make predictions
            
        Returns
        -------
        probas : np.ndarray of shape (n_samples, n_classes)
            Probabilities for each class
            
        Raises
        ------
        ValueError
            If called on a regression model.
        """
        if not self.trained:
            raise ValueError("The model must be trained before making predictions.")
        
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks.")
        
        X = np.array(X, dtype=float)
        activations, _ = self._forward_pass(X)
        return activations[-1]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the model score on the provided data.
        R² for regression, accuracy for classification.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test data
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            True values or labels
            
        Returns
        -------
        score : float
            R² score (regression) or accuracy (classification)
        """
        if not self.trained:
            raise ValueError("The model must be trained before calculating its score.")
        
        if self.task == 'classification':
            y_pred = self.predict(X)
            return np.mean(y_pred == y)
        else:  # regression
            y_true = np.array(y, dtype=float)
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
            
            y_pred = self.predict(X)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            u = ((y_true - y_pred) ** 2).sum()
            v = ((y_true - y_true.mean(axis=0)) ** 2).sum()
            
            if v == 0:
                return 0.0
            
            return 1 - u / v


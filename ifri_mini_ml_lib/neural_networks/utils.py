# Importations
import numpy as np
from typing import List, Tuple, Optional

from ifri_mini_ml_lib.preprocessing.preparation.splitting import DataSplitter

# Utility functions and globlals variables for neural networks

# Activation functions and their derivatives
def _leaky_relu( x: np.ndarray) -> np.ndarray:
    """
    Leaky ReLU activation function
    """
    return np.where(x > 0, x, 0.01 * x)

def _leaky_relu_derivative( x: np.ndarray) -> np.ndarray:
    """
    Derivative of Leaky ReLU function
    """
    return np.where(x > 0, 1, 0.01)

def _sigmoid( x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def _sigmoid_derivative( x: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid function
    """
    sigmoid_x = _sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def _softmax( x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def _softmax_derivative( x: np.ndarray) -> np.ndarray:
    """
    Derivative of softmax function
    For backpropagation with softmax and cross-entropy,
    this derivative is simplified and already handled in _backward_pass
    """
    s = _softmax(x)
    return s * (1 - s)

def _relu( x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function
    """
    return np.maximum(0, x)

def _relu_derivative( x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU function
    """
    return np.where(x > 0, 1, 0)

def _tanh( x: np.ndarray) -> np.ndarray:
    """
    Tanh activation function
    """
    return np.tanh(x)

def _tanh_derivative( x: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh function
    """
    return 1 - np.power(np.tanh(x), 2)

# Dictionary of activation functions and their derivatives

ACTIVATIONS = {
    'sigmoid': _sigmoid,
    'relu': _relu,
    'tanh': _tanh,
    'leaky_relu': _leaky_relu,
    'softmax': _softmax,
}

DERIVATIVES = {
    'sigmoid': _sigmoid_derivative,
    'relu': _relu_derivative,
    'tanh': _tanh_derivative,
    'leaky_relu': _leaky_relu_derivative,
    'softmax': _softmax_derivative,
}

# Allowed activations for regression and classification tasks

TASK_ACTIVATIONS = {
    "regression": {"sigmoid", "relu", "tanh", "leaky_relu"},
    "classification": {"sigmoid", "relu", "tanh", "leaky_relu", "softmax"},
}

# Weight initialization function

def initialize_weights(model, n_features: int, n_outputs: int) -> None:
    """
    Initialize the weights and biases of the network
    
    Parameters:
    -----------
    n_features : int
        Number of input features
    n_outputs : int
        Number of output classes
    """
    # Layer dimensions
    layer_sizes = [n_features] + list(model.hidden_layer_sizes) + [n_outputs]
    model.n_layers = len(layer_sizes) - 1
    model.n_outputs = n_outputs
    
    # Reset lists
    model.weights = []
    model.biases = []
    model.velocity_weights = []
    model.velocity_biases = []
    model.m_weights = []
    model.m_biases = []
    model.v_weights = []
    model.v_biases = []
    
    # Weight initialization with Xavier/Glorot method
    for i in range(model.n_layers):
        limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
        model.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
        model.biases.append(np.zeros(layer_sizes[i + 1]))
        
        # Initialization for optimizers
        model.velocity_weights.append(np.zeros_like(model.weights[-1]))  # For Momentum
        model.velocity_biases.append(np.zeros_like(model.biases[-1]))
        model.m_weights.append(np.zeros_like(model.weights[-1]))  # For Adam
        model.m_biases.append(np.zeros_like(model.biases[-1]))
        model.v_weights.append(np.zeros_like(model.weights[-1]))  # For Adam
        model.v_biases.append(np.zeros_like(model.biases[-1]))

# Weight update functions for different optimizers

def update_weights_sgd(model, gradients_w: List[np.ndarray], gradients_b: List[np.ndarray]) -> None:
    """
    Update weights with stochastic gradient descent
    """
    for i in range(model.n_layers):
        model.weights[i] -= model.learning_rate * gradients_w[i]
        model.biases[i] -= model.learning_rate * gradients_b[i]

def update_weights_momentum(model, gradients_w: List[np.ndarray], gradients_b: List[np.ndarray]) -> None:
    """
    Update weights with momentum gradient descent
    """
    for i in range(model.n_layers):
        model.velocity_weights[i] = model.momentum * model.velocity_weights[i] - model.learning_rate * gradients_w[i]
        model.velocity_biases[i] = model.momentum * model.velocity_biases[i] - model.learning_rate * gradients_b[i]
        
        model.weights[i] += model.velocity_weights[i]
        model.biases[i] += model.velocity_biases[i]

def update_weights_rmsprop(model, gradients_w: List[np.ndarray], gradients_b: List[np.ndarray]) -> None:
    """
    Update weights with RMSProp
    """
    decay_rate = 0.9
    
    for i in range(model.n_layers):
        # Update accumulators
        model.v_weights[i] = decay_rate * model.v_weights[i] + (1 - decay_rate) * np.square(gradients_w[i])
        model.v_biases[i] = decay_rate * model.v_biases[i] + (1 - decay_rate) * np.square(gradients_b[i])
        
        # Update weights
        model.weights[i] -= model.learning_rate * gradients_w[i] / (np.sqrt(model.v_weights[i] + model.epsilon))
        model.biases[i] -= model.learning_rate * gradients_b[i] / (np.sqrt(model.v_biases[i] + model.epsilon))

def update_weights_adam(model, gradients_w: List[np.ndarray], gradients_b: List[np.ndarray]) -> None:
    """
    Update weights with Adam optimizer
    """
    for i in range(model.n_layers):
        # Update moments
        model.m_weights[i] = model.beta1 * model.m_weights[i] + (1 - model.beta1) * gradients_w[i]
        model.m_biases[i] = model.beta1 * model.m_biases[i] + (1 - model.beta1) * gradients_b[i]
        
        # Update second moments
        model.v_weights[i] = model.beta2 * model.v_weights[i] + (1 - model.beta2) * np.square(gradients_w[i])
        model.v_biases[i] = model.beta2 * model.v_biases[i] + (1 - model.beta2) * np.square(gradients_b[i])
        
        # Bias correction
        m_weights_corrected = model.m_weights[i] / (1 - model.beta1 ** model.t)
        m_biases_corrected = model.m_biases[i] / (1 - model.beta1 ** model.t)
        v_weights_corrected = model.v_weights[i] / (1 - model.beta2 ** model.t)
        v_biases_corrected = model.v_biases[i] / (1 - model.beta2 ** model.t)
        
        # Update weights
        model.weights[i] -= model.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected + model.epsilon))
        model.biases[i] -= model.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected + model.epsilon))
    
    model.t += 1

def split_train_validation(model, X: np.ndarray, y: np.ndarray, seed = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    y : np.ndarray
        Target values or classes
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_val, y_train, y_val : The split datasets
    """
    splitter = DataSplitter(seed=seed)
    X_train, X_val, y_train, y_val = splitter.train_test_split(X, y, test_size=model.validation_fraction)
    
    return X_train, X_val, y_train, y_val

# Mapping of update methods for optimizers

UPDATE_WEIGHTS_METHODS = {
    'sgd': update_weights_sgd,
    'momentum': update_weights_momentum,
    'rmsprop': update_weights_rmsprop,
    'adam': update_weights_adam,
}
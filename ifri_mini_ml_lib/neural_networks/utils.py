# Importations
import numpy as np
from typing import List, Tuple, Optional

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

def _softmax(self, x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def _softmax_derivative(self, x: np.ndarray) -> np.ndarray:
    """
    Derivative of softmax function
    For backpropagation with softmax and cross-entropy,
    this derivative is simplified and already handled in _backward_pass
    """
    s = self._softmax(x)
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

def _tanh(self, x: np.ndarray) -> np.ndarray:
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

TASK_ACTIVATIONS = {
    "regression": {"sigmoid", "relu", "tanh", "leaky_relu"},
    "classification": {"sigmoid", "relu", "tanh", "leaky_relu", "softmax"},
}
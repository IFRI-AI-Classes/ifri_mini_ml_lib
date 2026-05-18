# Importations
import numpy as np
from typing import List, Tuple



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


def _linear( x: np.ndarray) -> np.ndarray:
    """
    Linear activation function 
    """
    return x

def _linear_derivative( x: np.ndarray) -> np.ndarray:
    """
    Derivative of linear function
    """
    return np.ones_like(x)

# Dictionary of activation functions and their derivatives

ACTIVATIONS = {
    'sigmoid': _sigmoid,
    'relu': _relu,
    'tanh': _tanh,
    'leaky_relu': _leaky_relu,
    'softmax': _softmax,
    'linear': _linear,
}

DERIVATIVES = {
    'sigmoid': _sigmoid_derivative,
    'relu': _relu_derivative,
    'tanh': _tanh_derivative,
    'leaky_relu': _leaky_relu_derivative,
    'softmax': _softmax_derivative,
    'linear': _linear_derivative,
}

# Allowed activations for regression and classification tasks
TASK_ACTIVATIONS = {
    "regression": {"sigmoid", "relu", "tanh", "leaky_relu"},
    "classification": {"sigmoid", "relu", "tanh", "leaky_relu", "softmax"},
}


class Activation:
    """
    Activation layer that applies a specified activation function and its derivative for backpropagation.
    """
    def __init__(self, f, f_prime):
        """
        Parameters:
        -----------
        f : function
            The activation function to apply in the forward pass
        f_prime : function
            The derivative of the activation function to apply in the backward pass
        """
        self.f = f
        self.f_prime = f_prime

    def forward(self, x):
        """
        Apply the activation function to the input x and store it for backpropagation
        Parameters:
        -----------
        x : np.ndarray
            The input to the activation function
        Returns:
        --------
        np.ndarray
            The output after applying the activation function   
        """
        self.x = x
        return self.f(x)

    def backward(self, grad_output):
        """Apply the derivative of the activation function to the incoming gradient
        Parameters:
        -----------
        grad_output : np.ndarray
            The gradient coming from the next layer during backpropagation
        Returns:
        --------
        np.ndarray
            The gradient to pass to the previous layer after applying the activation derivative
        """
        
        return grad_output * self.f_prime(self.x)
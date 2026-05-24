# Importations
import numpy as np
from typing import List, Tuple
from ..preprocessing.preparation.splitting import DataSplitter

# Utility functions and globlals variables for neural networks


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


# Mapping of update methods for optimizers

UPDATE_WEIGHTS_METHODS = {
    'sgd': update_weights_sgd,
    'momentum': update_weights_momentum,
    'rmsprop': update_weights_rmsprop,
    'adam': update_weights_adam,
}
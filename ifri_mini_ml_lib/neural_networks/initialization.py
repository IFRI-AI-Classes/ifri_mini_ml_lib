import numpy as np



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

import numpy as np

def euclidean_distance(x1, x2):
    """
    Computes the Euclidean distance between two vectors.

    Parameters:
    - x1 (numpy.ndarray): First vector.
    - x2 (numpy.ndarray): Second vector.

    Returns:
    - float: The Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2)**2))

import numpy as np

def euclidean_distance(x1, x2):
    """
    Description:
    ------------
    Computes the Euclidean distance between two vectors. Euclidean distance is the straight-line
    distance between two points in Euclidean space, calculated as the square root of the sum of
    the squared differences between corresponding elements of the vectors.

    Arguments:
    -----------
    - x1 (numpy.ndarray): First vector.
    - x2 (numpy.ndarray): Second vector.

    Functions:
    -----------
    - Calculates the element-wise difference between the two vectors.
    - Squares each element of the resulting vector.
    - Sums the squared elements.
    - Takes the square root of the sum.

    Returns:
    --------
    - float: The Euclidean distance between x1 and x2.

    Example:
    ---------
    distance = euclidean_distance(np.array([1, 2]), np.array([4, 6]))
    print(f"Euclidean distance: {distance}")
    """
    return np.sqrt(np.sum((x1 - x2)**2))

import numpy as np

def calculate_inertia(data, labels, centroids):
    """
    Description:
    ------------
    Computes the inertia of a clustering model.  Inertia measures the sum of squared distances
    from each data point to its cluster's centroid, providing an indication of cluster density
    and separation.  Lower inertia generally indicates better clustering.

    Arguments:
    -----------
    - data (numpy.ndarray): Input data array, shape (n_samples, n_features).
    - labels (numpy.ndarray): Cluster labels for each data point, shape (n_samples,).
    - centroids (numpy.ndarray): Cluster centers, shape (n_clusters, n_features).

    Functions:
    -----------
    - Iterates through each data point.
    - Finds the assigned centroid for the point.
    - Calculates the squared Euclidean distance between the point and its centroid.
    - Accumulates the distances to compute the total inertia.

    Returns:
    --------
    - float: Total inertia of the clustering.

    Example:
    ---------
    inertia = calculate_inertia(data, labels, centroids)
    print(f"Inertia: {inertia}")
    """
    inertia = 0
    for i in range(len(data)):
        centroid_idx = labels[i]
        centroid = centroids[centroid_idx]
        distance = np.linalg.norm(data[i] - centroid) ** 2
        inertia += distance
    
    return inertia



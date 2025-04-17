import numpy as np

def calculate_inertia(data, labels, centroids):
    """
    Computes the inertia of a clustering model.

    :param data: Input data (numpy array)
    :param labels: Cluster labels for each point (numpy array)
    :param centroids: Cluster centers (numpy array)
    :Return: Total inertia
    """
    inertia = 0
    for i in range(len(data)):
        centroid_idx = labels[i]
        centroid = centroids[centroid_idx]
        distance = np.linalg.norm(data[i] - centroid) ** 2
        inertia += distance
    
    return inertia




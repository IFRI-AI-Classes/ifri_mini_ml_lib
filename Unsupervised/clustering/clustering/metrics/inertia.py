import numpy as np

def calculate_inertia(data, labels, centroids):
    """
    Calcule l'inertie d'un modèle de clustering.
    
    :param data: Données d'entrée (numpy array)
    :param labels: Labels de cluster pour chaque point (numpy array)
    :param centroids: Centres des clusters (numpy array)
    :return: Inertie totale
    """
    inertia = 0
    for i in range(len(data)):
        centroid_idx = labels[i]
        centroid = centroids[centroid_idx]
        distance = np.linalg.norm(data[i] - centroid) ** 2
        inertia += distance
    
    return inertia




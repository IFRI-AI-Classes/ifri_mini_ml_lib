import numpy as np
from clustering.utils.utils import euclidean_distance

def calculate_silhouette(data, labels):
    """
    Calcule le score silhouette pour un modèle de clustering.
    
    :param data: Données d'entrée (numpy array)
    :param labels: Labels de cluster pour chaque point (numpy array)
    :return: Score silhouette moyen
    """
    if len(np.unique(labels)) > 1:
        silhouette_scores = []
        for i in range(len(data)):
            # Calculer la distance moyenne au sein du même cluster (a)
            cluster_label = labels[i]
            cluster_points = data[labels == cluster_label]
            if len(cluster_points) > 1:
                distances_same_cluster = [euclidean_distance(data[i], point) for point in cluster_points if not np.array_equal(point, data[i])]
                a = np.mean(distances_same_cluster)
            else:
                a = 0  # Si un seul point dans le cluster

            # Trouver le cluster le plus proche (b)
            other_clusters = [label for label in np.unique(labels) if label != cluster_label]
            b_values = []
            for other_label in other_clusters:
                other_cluster_points = data[labels == other_label]
                distances_other_cluster = [euclidean_distance(data[i], point) for point in other_cluster_points]
                b_values.append(np.mean(distances_other_cluster))

            if b_values:
                b = np.min(b_values)
            else:
                b = a  # Si pas d'autres clusters

            # Calculer le score silhouette
            if a == b:
                silhouette = 0
            elif a < b:
                silhouette = (b - a) / max(a, b)
            else:
                silhouette = (b - a) / a

            silhouette_scores.append(silhouette)

        return np.mean(silhouette_scores)
    else:
        return 0  # Un seul cluster, pas de séparation


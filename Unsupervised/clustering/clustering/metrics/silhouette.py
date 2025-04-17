import numpy as np
from clustering.utils.utils import euclidean_distance

def calculate_silhouette(data, labels):
    """
    Computes the silhouette score for a clustering model.

   :param data: Input data (numpy array)
   :param labels: Cluster labels for each point (numpy array)
   :return: Average silhouette score
   """
    if len(np.unique(labels)) > 1:
        silhouette_scores = []
        for i in range(len(data)):
            # Calculate the average distance within the same cluster (a)
            cluster_label = labels[i]
            cluster_points = data[labels == cluster_label]
            if len(cluster_points) > 1:
                distances_same_cluster = [euclidean_distance(data[i], point) for point in cluster_points if not np.array_equal(point, data[i])]
                a = np.mean(distances_same_cluster)
            else:
                a = 0  # If only one point in the cluster

            # Find the nearest cluster (b)
            other_clusters = [label for label in np.unique(labels) if label != cluster_label]
            b_values = []
            for other_label in other_clusters:
                other_cluster_points = data[labels == other_label]
                distances_other_cluster = [euclidean_distance(data[i], point) for point in other_cluster_points]
                b_values.append(np.mean(distances_other_cluster))

            if b_values:
                b = np.min(b_values)
            else:
                b = a  # if no other clusters

            # Calculate silhouette score
            if a == b:
                silhouette = 0
            elif a < b:
                silhouette = (b - a) / max(a, b)
            else:
                silhouette = (b - a) / a

            silhouette_scores.append(silhouette)

        return np.mean(silhouette_scores)
    else:
        return 0  # One cluster, no separation

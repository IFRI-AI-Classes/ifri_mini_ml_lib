import numpy as np

from ifri_mini_ml_lib.utils.tools import _euclidean_distance


def calculate_inertia(data, labels, centroids):
    """
    Computes the inertia of a clustering model.  Inertia measures the sum of squared distances
    from each data point to its cluster's centroid, providing an indication of cluster density
    and separation.  Lower inertia generally indicates better clustering.

    Arguments:
        data (numpy.ndarray): Input data array, shape (n_samples, n_features).
        labels (numpy.ndarray): Cluster labels for each data point, shape (n_samples,).
        centroids (numpy.ndarray): Cluster centers, shape (n_clusters, n_features).

    Returns:
        float: Total inertia of the clustering.

    Example:
        >>> inertia = calculate_inertia(data, labels, centroids)
        >>> print(f"Inertia: {inertia}")
    """
    inertia = 0
    for i in range(len(data)):
        centroid_idx = labels[i]
        centroid = centroids[centroid_idx]
        distance = np.linalg.norm(data[i] - centroid) ** 2
        inertia += distance
    return inertia


def calculate_silhouette(data, labels):
    """
    Computes the average silhouette score for a clustering model. The silhouette score measures
    how well each data point fits within its assigned cluster compared to other clusters.
    It ranges from -1 to 1, where higher values indicate better clustering.

    Arguments:
        data (numpy.ndarray): Input data array, shape (n_samples, n_features).
        labels (numpy.ndarray): Cluster labels for each data point, shape (n_samples,).

    Returns:
        float: Average silhouette score.

    Example:
    >>> silhouette_avg = calculate_silhouette(data, labels)
    >>> print(f"Silhouette Score: {silhouette_avg}")
    """
    if len(np.unique(labels)) > 1:
        silhouette_scores = []
        for i in range(len(data)):
            # Calculate the average distance within the same cluster (a)
            cluster_label = labels[i]
            cluster_points = data[labels == cluster_label]
            if len(cluster_points) > 1:
                distances_same_cluster = [_euclidean_distance(data[i], point) for point in cluster_points if not np.array_equal(point, data[i])]
                a = np.mean(distances_same_cluster)
            else:
                a = 0  # If only one point in the cluster

            # Find the nearest cluster (b)
            other_clusters = [label for label in np.unique(labels) if label != cluster_label]
            b_values = []
            for other_label in other_clusters:
                other_cluster_points = data[labels == other_label]
                distances_other_cluster = [_euclidean_distance(data[i], point) for point in other_cluster_points]
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

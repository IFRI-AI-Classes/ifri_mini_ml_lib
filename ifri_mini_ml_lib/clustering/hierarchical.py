import numpy as np
from .utils import euclidean_distance  # Import function euclidean_distance
from matplotlib import pyplot as plt
from .kmeans import KMeans

class HierarchicalClustering:
    """
    Description:
    ------------
    HierarchicalClustering performs hierarchical clustering using either the agglomerative (bottom-up)
    or divisive (top-down) approach.

    Arguments:
        n_clusters (int or None): Desired number of clusters (required for divisive method).
        linkage (str): Linkage criterion to use for merging clusters ('single', 'complete', 'average').
        method (str): Clustering strategy ('agglomerative' or 'divisive').

    Example:
    >>> from ifri_mini_ml_lib.clustering import HierarchicalClustering
    >>> hierarchical = HierarchicalClustering(n_clusters=3, linkage='complete', method='agglomerative')
    >>> labels = hierarchical.fit_predict(data)
    >>> hierarchical.plot_clusters(data, labels)
    """

    def __init__(self, n_clusters=None, linkage='single', method='agglomerative'):
        """
        Initializes the hierarchical clustering parameters.

        Arguments:
            n_clusters (int, optional): Desired number of clusters (required for the divisive method).
                If None, agglomerative clustering proceeds until all points are merged.
            linkage (str, optional): Linkage criterion to use ('single', 'complete', 'average').
            method (str, optional): Clustering method to apply ('agglomerative' or 'divisive').
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.method = method
        self.labels = None
        self.linked_matrix = None  # Adding a variable to store the linkage matrix

    def fit_predict(self, data, kmeans=None):
        """
        Performs hierarchical clustering on the given data.

        Arguments:
            data (numpy.ndarray): Data to be clustered (n_samples, n_features).
            kmeans (KMeans, optional): KMeans instance, required for the divisive method
        """
        if self.method == 'agglomerative':
            self.labels = self._agglomerative_clustering(data)
        elif self.method == 'divisive':
            if self.n_clusters is None:
                raise ValueError("n_clusters must be specified for the divisive method.")
            self.labels = self._divisive_clustering(data, kmeans)
        else:
            raise ValueError("Method not found. Choose 'agglomerative' or 'divisive'.")

        return self.labels

    def _agglomerative_clustering(self, data):
        """
        Implements agglomerative (bottom-up) hierarchical clustering.
        """
        # Initialize each data point as a separate cluster (singleton)
        clusters = [{i} for i in range(len(data))]
    
        # Compute the pairwise distance matrix between all points
        distances = self._compute_distance_matrix(data)
    
        n_samples = len(data)
        # Track the current cluster indices (to assign unique indices to merged clusters)
        cluster_indices = list(range(n_samples))
        # Will store the linkage matrix: each row = [cluster1, cluster2, distance, new_cluster_size]
        linkage_matrix = []
        # Next cluster index to assign to a newly merged cluster
        next_cluster_idx = n_samples

        # Main loop: continue until the desired number of clusters is reached
        while len(clusters) > self.n_clusters if self.n_clusters else len(clusters) > 1:
            # Find the two closest clusters
            min_i, min_j = None, None
            min_distance = float('inf')
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Compute the linkage distance between clusters i and j
                    distance = self._compute_linkage_distance(clusters[i], clusters[j], distances, self.linkage)
                    if distance < min_distance:
                        min_distance = distance
                        min_i, min_j = i, j

            # Record the merge in the linkage matrix
            c1_idx = cluster_indices[min_i]
            c2_idx = cluster_indices[min_j]
            new_cluster_size = len(clusters[min_i]) + len(clusters[min_j])
            linkage_matrix.append([c1_idx, c2_idx, min_distance, new_cluster_size])

            # Merge the two closest clusters
            clusters[min_i] = clusters[min_i].union(clusters[min_j])
            del clusters[min_j]
            # Update cluster indices: assign a new index to the merged cluster, remove the merged one
            cluster_indices[min_i] = next_cluster_idx
            del cluster_indices[min_j]
            next_cluster_idx += 1

            # If the desired number of clusters is reached, exit the loop
            if self.n_clusters and len(clusters) <= self.n_clusters:
                break

        # Store the linkage matrix as a numpy array (for dendrogram plotting)
        self.linkage_matrix = np.array(linkage_matrix)

        # Assign labels: for each cluster, assign the same label to all its points
        labels = np.zeros(len(data), dtype=int)
        for i, cluster in enumerate(clusters):
            for point_index in cluster:
                labels[point_index] = i

        return labels

    def _divisive_clustering(self, data, kmeans):
        """
        Implements divisive (top-down) hierarchical clustering.

        Arguments:
            data (numpy.ndarray): The dataset to be clustered.
            kmeans (KMeans): An instance of a KMeans algorithm used to split clusters.

        Returns:
            numpy.ndarray: Cluster labels assigned to each data point.
        """
        # Initialization: all points are in a single cluster
        clusters = [set(range(len(data)))]

        while len(clusters) < self.n_clusters:
            # Find the largest cluster (the one with the greatest number of points)
            largest_cluster_index = np.argmax([len(cluster) for cluster in clusters])
            largest_cluster = clusters[largest_cluster_index]

            # Split the larger cluster into two sub-clusters
            cluster1, cluster2 = self._bisect_cluster(data, largest_cluster, kmeans)

            # Replace the original cluster with the two new subclusters
            del clusters[largest_cluster_index]
            clusters.append(cluster1)
            clusters.append(cluster2)

        # Assign labels
        labels = np.zeros(len(data), dtype=int)
        for i, cluster in enumerate(clusters):
            for point_index in cluster:
                labels[point_index] = i

        return labels

    def _bisect_cluster(self, data, cluster, kmeans):
        """
        Splits a cluster into two sub-clusters using a simple method (K-Means with k=2).

        Arguments:
            data (numpy.ndarray): The complete dataset.
            cluster (set): Indices of the data points belonging to the cluster to be split.
            kmeans (KMeans): An instance of the KMeans algorithm (can be reinitialized inside the method).

        Returns:
            tuple: Two sets representing the indices of the resulting sub-clusters.
        """
        # Convert cluster to data
        cluster_data = data[list(cluster)]

        # Use k-means to divide the cluster into two
        kmeans = KMeans(n_clusters=2, random_state=42)  # Vous pouvez ajuster les paramètres de KMeans
        labels = kmeans.fit_predict(cluster_data)

        # Create the two subclusters
        cluster1 = set([list(cluster)[i] for i in range(len(cluster)) if labels[i] == 0])
        cluster2 = set([list(cluster)[i] for i in range(len(cluster)) if labels[i] == 1])

        return cluster1, cluster2

    def _compute_distance_matrix(self, data):
        """
        Description:
        ------------
        Computes the distance matrix between all data points.

        Arguments:
        -----------
        - data (numpy.ndarray): The dataset (n_samples, n_features).

        Returns:
        --------
        - numpy.ndarray: A symmetric matrix containing pairwise distances between data points.
        """
        n_samples = len(data)
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = distances[j, i] = euclidean_distance(data[i], data[j])
        return distances

    def _compute_linkage_distance(self, cluster1, cluster2, distances, linkage='single'):
        """
        Description:
        ------------
        Computes the distance between two clusters using the specified linkage criterion.

        Arguments:
        -----------
        - cluster1 (set): Indices of the first cluster.
        - cluster2 (set): Indices of the second cluster.
        - distances (numpy.ndarray): Precomputed distance matrix between all data points.
        - linkage (str): Linkage criterion to use ('single', 'complete', or 'average').

        Returns:
        --------
        - float: The computed linkage distance between the two clusters.
        """
        if linkage == 'single':
            # Minimum distance between points of the two clusters
            min_distance = float('inf')
            for i in cluster1:
                for j in cluster2:
                    distance = distances[i, j]
                    if distance < min_distance:
                        min_distance = distance
            return min_distance
        elif linkage == 'complete':
            # Maximum distance between points of the two clusters
            max_distance = float('-inf')
            for i in cluster1:
                for j in cluster2:
                    distance = distances[i, j]
                    if distance > max_distance:
                        max_distance = distance
            return max_distance
        elif linkage == 'average':
            # Average distance between points of the two clusters
            total_distance = 0
            for i in cluster1:
                for j in cluster2:
                    total_distance += distances[i, j]
            return total_distance / (len(cluster1) * len(cluster2))
        else:
            raise ValueError("Critère de linkage non reconnu. Choisissez 'single', 'complete' ou 'average'.")
        

    def plot_dendrogram(self, labels=None):
        """
        Plots a dendrogram based on the linkage matrix constructed during hierarchical clustering.
        The main branches corresponding to the final clusters are colored differently for better visualization.

        Arguments:
            data (numpy.ndarray): The dataset used for clustering.

        Example:
        >>> hierarchical = HierarchicalClustering(n_clusters=3, linkage='complete', method='agglomerative')
        >>> labels = hierarchical.fit_predict(data)
        >>> hierarchical.plot_dendrogram(data)
        """
        if self.method != 'agglomerative':
            raise NotImplementedError("Dendrogram plotting is only implemented for agglomerative clustering.")

        max_labels = 100  # Maximum number of labels to display on the leaves

        # Check if the linkage matrix exists; if not, raise an error
        if not hasattr(self, "linkage_matrix") or self.linkage_matrix is None:
            raise ValueError("The linkage matrix is not available. Run fit_predict first.")

        linkage_matrix = self.linkage_matrix
        n_samples = linkage_matrix.shape[0] + 1

        # If labels are not provided, use sample indices as labels
        if labels is None:
            labels = [str(i) for i in range(n_samples)]

        # Option: show leaf labels only if there are not too many samples
        show_labels = n_samples <= max_labels

        # Determine the main clusters at the cut
        # Take the n_clusters largest clusters formed at the end
        n_clusters = self.n_clusters if self.n_clusters else 2
        cluster_ids = list(range(n_samples, n_samples + linkage_matrix.shape[0]))
        # The main clusters are those which remain at the end
        main_clusters = set(cluster_ids[-n_clusters+1:]) if n_clusters > 1 else set([cluster_ids[-1]])

        # Color palette for cluster branches
        import matplotlib.cm as cm
        colors = plt.colormaps.get_cmap('tab10')
        branch_colors = {}

        # Initialize horizontal (x) and vertical (y) positions for leaves
        x_pos = {i: i for i in range(n_samples)}
        y_pos = {i: 0 for i in range(n_samples)}

        # Create the plot with a dynamic width depending on the number of samples
        plt.figure(figsize=(max(8, min(20, n_samples // 5)), 6))

        # Draw the branches of the dendrogram
        for k, (c1, c2, dist, count) in enumerate(linkage_matrix):
            c1, c2 = int(c1), int(c2)
            # Protect against KeyError if a cluster index is missing
            if c1 not in x_pos or c2 not in x_pos:
                continue
            x1, x2 = x_pos[c1], x_pos[c2]
            y1, y2 = y_pos[c1], y_pos[c2]
            x_new = (x1 + x2) / 2
            cluster_idx = n_samples + k

            # Color the main branches (clusters at the cut)
            if cluster_idx in main_clusters:
                color = colors(len(branch_colors) % colors.N)
                branch_colors[cluster_idx] = color
            else:
                # Inherit the color from the parent branch if possible, otherwise use gray
                color = branch_colors.get(c1, branch_colors.get(c2, "#888888"))

            # Draw the vertical and horizontal lines for the current merge
            plt.plot([x1, x1], [y1, dist], c=color)
            plt.plot([x2, x2], [y2, dist], c=color)
            plt.plot([x1, x2], [dist, dist], c=color)
            # Store the position of the new cluster
            x_pos[cluster_idx] = x_new
            y_pos[cluster_idx] = dist

        # Display leaf labels if not too many samples
        if show_labels:
            for i in range(n_samples):
                plt.text(x_pos[i], -0.01, labels[i], ha='right', va='top', rotation=90, fontsize=8)

        plt.xlabel("Sample index")
        plt.ylabel("Distance")
        plt.title("Dendrogram")
        plt.tight_layout()
        plt.show()


    def plot_clusters(self, data):
        """
        Plots a scatter plot of the data points colored by their cluster labels,
        supporting 1D, 2D, and 3D data. For higher dimensions, PCA is applied to reduce to 3D.

        Arguments:
            data (numpy.ndarray): Data array.

        Example:
        >>> hierarchical.plot_clusters(data)
        """
        n_features = data.shape[1]
        labels = self.fit_predict(data)
        if n_features > 3:
            from ifri_mini_ml_lib.preprocessing.dimensionality_reduction.pca import PCA
            pca = PCA(n_component=3)
            data = pca.fit_transform(data)
            n_features = 3
            print("Warning: Data reduced to 3D using PCA for visualization.")

        unique_labels = np.unique(labels)
        colors = plt.colormaps.get_cmap('tab10')
        fig = plt.figure(figsize=(8, 6))

        # Case 1D
        if n_features == 1:
            ax = fig.add_subplot(111)
            for i, label in enumerate(unique_labels):
                cluster_points = data[labels == label]
                ax.scatter(cluster_points[:, 0], np.zeros_like(cluster_points[:, 0]),
                        label=f"Cluster {label}", color=colors(i), s=50)
            ax.set_yticks([])
            ax.set_xlabel("Feature 1")
            ax.set_title("Hierarchical Clustering Result (1D)")

        # Case 2D
        elif n_features == 2:
            ax = fig.add_subplot(111)
            for i, label in enumerate(unique_labels):
                cluster_points = data[labels == label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        label=f"Cluster {label}", color=colors(i), s=50)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_title("Hierarchical Clustering Result (2D)")

        # Case 3D
        elif n_features == 3:
            ax = fig.add_subplot(111, projection='3d')
            for i, label in enumerate(unique_labels):
                cluster_points = data[labels == label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                           label=f"Cluster {label}", color=colors(i), s=50)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
            ax.set_title("Hierarchical Clustering Result (3D)")

        plt.legend()
        plt.grid(True)
        plt.show()

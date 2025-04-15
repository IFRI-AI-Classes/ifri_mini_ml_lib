import numpy as np
from utils.utils import euclidean_distance  # Importez la fonction euclidean_distance
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from clustering.algo.kmeans import KMeans

class HierarchicalClustering:
    """
    Implémentation du clustering hiérarchique avec les méthodes agglomérative et descendante.
    """

    def __init__(self, n_clusters=None, linkage='single', method='agglomerative'):
        """
        Initialise les paramètres du clustering hiérarchique.

        Args:
            n_clusters (int, optional): Le nombre de clusters à la fin (pour la méthode descendante).
                                        Si None, l'algorithme s'arrête quand tous les points sont dans un cluster (agglomérative).
            linkage (str, optional): Le critère de linkage à utiliser ('single', 'complete', 'average').
            method (str, optional): La méthode à utiliser ('agglomerative', 'divisive').
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.method = method
        self.labels = None
        self.linked_matrix = None  # Ajout d'une variable pour stocker la matrice de linkage

    def fit_predict(self, data, kmeans=None):
        """
        Effectue le clustering hiérarchique sur les données fournies.

        Args:
            data (numpy.ndarray): Les données à clusteriser (n_samples, n_features).

        Returns:
            numpy.ndarray: Les étiquettes de cluster pour chaque point.
        """
        if self.method == 'agglomerative':
            self.labels = self._agglomerative_clustering(data)
        elif self.method == 'divisive':
            if self.n_clusters is None:
                raise ValueError("n_clusters doit être spécifié pour la méthode divisive.")
            self.labels = self._divisive_clustering(data, kmeans)
        else:
            raise ValueError("Méthode non reconnue. Choisissez 'agglomerative' ou 'divisive'.")

        return self.labels

    def _agglomerative_clustering(self, data):
        """
        Implémente le clustering hiérarchique agglomératif (ascendant).
        """
        # Initialisation : chaque point est un cluster
        clusters = [{i} for i in range(len(data))]
        distances = self._compute_distance_matrix(data)

        while len(clusters) > 1:
            # Trouver les deux clusters les plus proches
            min_i, min_j = None, None
            min_distance = float('inf')
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._compute_linkage_distance(clusters[i], clusters[j], distances, self.linkage)
                    if distance < min_distance:
                        min_distance = distance
                        min_i, min_j = i, j

            # Fusionner les deux clusters les plus proches
            clusters[min_i] = clusters[min_i].union(clusters[min_j])
            del clusters[min_j]

        # Assigner les labels
        labels = np.zeros(len(data), dtype=int)
        for i, cluster in enumerate(clusters):
            for point_index in cluster:
                labels[point_index] = i

        return labels

    def _divisive_clustering(self, data, kmeans):
        """
        Implémente le clustering hiérarchique divisif (descendant).
        """
        # Initialisation : tous les points sont dans un seul cluster
        clusters = [set(range(len(data)))]

        while len(clusters) < self.n_clusters:
            # Trouver le cluster le plus grand (celui avec le plus de points)
            largest_cluster_index = np.argmax([len(cluster) for cluster in clusters])
            largest_cluster = clusters[largest_cluster_index]

            # Diviser le cluster le plus grand en deux sous-clusters
            cluster1, cluster2 = self._bisect_cluster(data, largest_cluster, kmeans)

            # Remplacer le cluster original par les deux nouveaux sous-clusters
            del clusters[largest_cluster_index]
            clusters.append(cluster1)
            clusters.append(cluster2)

        # Assigner les labels
        labels = np.zeros(len(data), dtype=int)
        for i, cluster in enumerate(clusters):
            for point_index in cluster:
                labels[point_index] = i

        return labels

    def _bisect_cluster(self, data, cluster, kmeans):
        """
        Divise un cluster en deux sous-clusters (méthode simple : k-means avec k=2).
        """
        # Convertir le cluster en données
        cluster_data = data[list(cluster)]

        # Utiliser k-means pour diviser le cluster en deux
        kmeans = KMeans(n_clusters=2, random_state=42)  # Vous pouvez ajuster les paramètres de KMeans
        labels = kmeans.fit_predict(cluster_data)

        # Créer les deux sous-clusters
        cluster1 = set([list(cluster)[i] for i in range(len(cluster)) if labels[i] == 0])
        cluster2 = set([list(cluster)[i] for i in range(len(cluster)) if labels[i] == 1])

        return cluster1, cluster2

    def _compute_distance_matrix(self, data):
        """
        Calcule la matrice de distance entre tous les points.
        """
        n_samples = len(data)
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = distances[j, i] = euclidean_distance(data[i], data[j])
        return distances

    def _compute_linkage_distance(self, cluster1, cluster2, distances, linkage='single'):
        """
        Calcule la distance entre deux clusters en utilisant le critère de linkage spécifié.
        """
        if linkage == 'single':
            # Distance minimale entre les points des deux clusters
            min_distance = float('inf')
            for i in cluster1:
                for j in cluster2:
                    distance = distances[i, j]
                    if distance < min_distance:
                        min_distance = distance
            return min_distance
        elif linkage == 'complete':
            # Distance maximale entre les points des deux clusters
            max_distance = float('-inf')
            for i in cluster1:
                for j in cluster2:
                    distance = distances[i, j]
                    if distance > max_distance:
                        max_distance = distance
            return max_distance
        elif linkage == 'average':
            # Distance moyenne entre les points des deux clusters
            total_distance = 0
            for i in cluster1:
                for j in cluster2:
                    total_distance += distances[i, j]
            return total_distance / (len(cluster1) * len(cluster2))
        else:
            raise ValueError("Critère de linkage non reconnu. Choisissez 'single', 'complete' ou 'average'.")

    def plot_dendrogram(self, data, **kwargs):
        """
        Génère et affiche le dendrogramme.

        Args:
            data (numpy.ndarray): Les données utilisées pour le clustering.
            **kwargs: Arguments additionnels à passer à la fonction dendrogram de scipy.
        """
        # Calculer la matrice de linkage
        if self.method == 'agglomerative':
            # Clustering hiérarchique agglomératif
            self.linked_matrix = linkage(data, method=self.linkage)

            # Dessiner le dendrogramme
            plt.figure(figsize=(12, 6))
            dendrogram(self.linked_matrix, orientation='top', **kwargs)
            plt.title('Dendrogramme Hierarchical Clustering')
            plt.xlabel('Samples')
            plt.ylabel('Distance')
            plt.show()

        elif self.method == 'divisive':
            # Clustering hiérarchique divisif
            print("Dendrogramme non supporté pour la méthode divisive.")
            return

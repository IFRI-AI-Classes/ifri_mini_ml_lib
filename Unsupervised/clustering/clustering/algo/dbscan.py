import numpy as np
from utils.utils import euclidean_distance  # Importez la fonction euclidean_distance

class DBSCAN:
    """
    Implémentation de l'algorithme DBSCAN pour le clustering.
    """

    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialise les paramètres de DBSCAN.

        Args:
            eps (float): Le rayon maximal pour considérer deux points comme voisins.
            min_samples (int): Le nombre minimal de points pour former un cluster.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None  # Étiquettes des clusters

    def fit_predict(self, data):
        """
        Effectue le clustering DBSCAN sur les données fournies.

        Args:
            data (numpy.ndarray): Les données à clusteriser (n_samples, n_features).

        Returns:
            numpy.ndarray: Les étiquettes de cluster pour chaque point (-1 pour le bruit).
        """
        self.labels = np.full(len(data), -1)  # Initialise tous les points comme bruit
        cluster_id = 0

        for i in range(len(data)):
            if self.labels[i] != -1:
                continue  # Point déjà visité

            # Trouve les voisins du point courant
            neighbors = self._region_query(data, i)

            if len(neighbors) < self.min_samples:
                # Pas un point central, reste bruit
                continue

            # Nouveau cluster
            self._expand_cluster(data, i, cluster_id, neighbors)
            cluster_id += 1

        return self.labels

    def _region_query(self, data, point_index):
        """
        Trouve les voisins d'un point dans un rayon donné.

        Args:
            data (numpy.ndarray): Les données.
            point_index (int): L'index du point.

        Returns:
            list: Les indices des voisins.
        """
        neighbors = []
        for i in range(len(data)):
            if euclidean_distance(data[point_index], data[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, data, point_index, cluster_id, neighbors):
        """
        Étend un cluster à partir d'un point central.

        Args:
            data (numpy.ndarray): Les données.
            point_index (int): L'index du point central.
            cluster_id (int): L'ID du cluster courant.
            neighbors (list): Les indices des voisins du point central.
        """
        self.labels[point_index] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_index = neighbors[i]
            if self.labels[neighbor_index] == -1:
                # Visite le voisin
                self.labels[neighbor_index] = cluster_id

                # Trouve les voisins du voisin
                new_neighbors = self._region_query(data, neighbor_index)
                if len(new_neighbors) >= self.min_samples:
                    # Ajoute les nouveaux voisins à la liste des voisins à visiter
                    neighbors += new_neighbors
            i += 1

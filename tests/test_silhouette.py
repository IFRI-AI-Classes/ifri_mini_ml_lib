import numpy as np
import pytest
from ifri_mini_ml_lib.metrics.evaluation_metrics import calculate_silhouette

@pytest.fixture
def sample_data():
    """ Jeu de données simple avec des points regroupés en deux clusters. """
    return np.array([
        [1, 2], [1, 3], [2, 2],  # Cluster 0
        [8, 8], [8, 9], [9, 8],  # Cluster 1
        [50, 50]                 # Bruit (point éloigné)
    ])

@pytest.fixture
def sample_labels():
    """ Labels correspondant aux clusters définis. """
    return np.array([0, 0, 0, 1, 1, 1, 2])  # Le dernier point est un cluster séparé

def test_calculate_silhouette(sample_data, sample_labels):
    """ Vérifie que le score de silhouette est correctement calculé. """
    silhouette_score = calculate_silhouette(sample_data, sample_labels)

    # Vérifier que le score est compris entre -1 et 1
    assert -1 <= silhouette_score <= 1

    # Vérifier que le score est positif pour un bon clustering
    assert silhouette_score > 0  

def test_single_cluster():
    """ Vérifie que le score est 0 si un seul cluster est présent. """
    single_cluster_data = np.array([
        [1, 1], [2, 2], [3, 3]
    ])
    single_cluster_labels = np.array([0, 0, 0])  # Tous les points dans un seul cluster

    silhouette_score = calculate_silhouette(single_cluster_data, single_cluster_labels)

    # Le score doit être 0 car il n'y a pas de séparation
    assert silhouette_score == 0

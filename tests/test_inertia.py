import numpy as np
import pytest
from ifri_mini_ml_lib.metrics.evaluation_metrics import calculate_inertia

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

@pytest.fixture
def sample_centroids():
    """ Centroïdes théoriques des clusters. """
    return np.array([
        [1.5, 2.5],  # Centre du cluster 0
        [8.5, 8.5],  # Centre du cluster 1
        [50, 50]     # Point bruit isolé
    ])

def test_calculate_inertia(sample_data, sample_labels, sample_centroids):
    """ Vérifie que l'inertie est correctement calculée et suit une logique attendue. """
    inertia = calculate_inertia(sample_data, sample_labels, sample_centroids)

    # Vérifier que l'inertie est une valeur positive
    assert inertia > 0

    # Vérifier qu'en réduisant la distance intra-cluster, l'inertie diminue
    adjusted_centroids = np.array([
        [1, 2],  # Rapprochement des points
        [8, 8],  # Réduction des écarts
        [50, 50] # Outlier inchangé
    ])
    lower_inertia = calculate_inertia(sample_data, sample_labels, adjusted_centroids)
    assert lower_inertia > inertia  # Meilleur regroupement = inertie plus faible

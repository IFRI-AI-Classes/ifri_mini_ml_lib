import numpy as np
import pytest
from ifri_mini_ml_lib.clustering.hierarchical import HierarchicalClustering
from ifri_mini_ml_lib.clustering.kmeans import KMeans

@pytest.fixture
def sample_data():
    """ Jeu de données simple avec 2 clusters bien séparés et un point bruit. """
    return np.array([
        [1, 2], [1, 3], [2, 2],      # Cluster 0
        [8, 8], [8, 9], [9, 8],      # Cluster 1
        [50, 50]                     # Bruit (pour tester la robustesse)
    ])

def test_init():
    """ Teste l'initialisation de HierarchicalClustering. """
    model = HierarchicalClustering(n_clusters=2, method='agglomerative', linkage='single')
    assert model.n_clusters == 2
    assert model.method == 'agglomerative'
    assert model.linkage == 'single'
    assert model.labels is None

def test_fit_predict_agglomerative(sample_data):
    """ Teste l'algorithme agglomératif pour le clustering hiérarchique. """
    model = HierarchicalClustering(n_clusters=2, method='agglomerative', linkage='single')
    labels = model.fit_predict(sample_data)
    # Vérifier que nous avons bien 2 clusters
    assert len(set(labels)) == 2  

    # Vérifier que les six premiers points appartiennent au même cluster
    assert len(set(labels[:6])) == 1  

    # Vérifier que le dernier point appartient à un cluster différent
    assert labels[-1] not in set(labels[:6])


def test_fit_predict_divisive(sample_data):
    """ Teste l'algorithme divisif pour le clustering hiérarchique. """
    kmeans = KMeans(n_clusters=2, random_state=42)  # Utilisation de KMeans pour la division
    model = HierarchicalClustering(n_clusters=2, method='divisive')
    labels = model.fit_predict(sample_data, kmeans=kmeans)
    # Vérifier que nous avons bien 2 clusters
    assert len(set(labels)) == 2  

    # Vérifier que les six premiers points appartiennent au même cluster
    assert len(set(labels[:6])) == 1  

    # Vérifier que le dernier point appartient à un cluster différent
    assert labels[-1] not in set(labels[:6])


def test_plot_dendrogram_runs_without_error(sample_data):
    """ Vérifie que la génération du dendrogramme ne produit pas d'erreur. """
    model = HierarchicalClustering(n_clusters=2, method='agglomerative', linkage='single')
    model.fit_predict(sample_data)
    model.plot_dendrogram(sample_data)

def test_plot_clusters_runs_without_error(sample_data):
    """ Vérifie que la fonction `plot_clusters` s'exécute sans erreur. """
    model = HierarchicalClustering(n_clusters=2, method='agglomerative', linkage='single')
    model.fit_predict(sample_data)
    model.plot_clusters(sample_data)

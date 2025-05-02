import numpy as np
import pytest
from ifri_mini_ml_lib.clustering.dbscan import DBSCAN

@pytest.fixture
def sample_data():
    # Jeu de données simple 2D avec 2 clusters distincts et un point bruit
    return np.array([
        [1, 2], [1, 3], [2, 2],      # Cluster 0
        [8, 8], [8, 9], [9, 8],      # Cluster 1
        [50, 50]                     # Bruit
    ])

def test_init():
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    assert dbscan.eps == 0.5
    assert dbscan.min_samples == 2
    assert dbscan.labels is None

def test_region_query(sample_data):
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    neighbors = dbscan._region_query(sample_data, 0)
    # Le point 0 doit avoir au moins lui-même et les points proches (1,2)
    assert 0 in neighbors
    assert 1 in neighbors
    assert 2 in neighbors
    # Le point bruit ne doit pas être dans les voisins
    assert 6 not in neighbors

def test_expand_cluster(sample_data):
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    dbscan.labels = np.full(len(sample_data), -1)
    neighbors = dbscan._region_query(sample_data, 0)
    dbscan._expand_cluster(sample_data, 0, cluster_id=0, neighbors=neighbors)
    # Les points du cluster 0 doivent avoir le label 0
    assert all(dbscan.labels[i] == 0 for i in [0,1,2])
    # Le reste doit rester à -1
    assert all(dbscan.labels[i] == -1 for i in [3,4,5,6])

def test_fit_predict(sample_data):
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    labels = dbscan.fit_predict(sample_data)
    # On doit avoir 2 clusters (labels 0 et 1) et un bruit (-1)
    assert set(labels) == {0, 1, -1}
    # Vérifier que les points des clusters sont bien groupés
    cluster_0 = [0,1,2]
    cluster_1 = [3,4,5]
    for i in cluster_0:
        assert labels[i] == 0
    for i in cluster_1:
        assert labels[i] == 1
    # Le point bruit
    assert labels[6] == -1

def test_plot_clusters_runs_without_error(sample_data):
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    dbscan.fit_predict(sample_data)
    # Juste vérifier que la fonction plot_clusters s'exécute sans erreur
    dbscan.plot_clusters(sample_data)

import numpy as np
import pytest

from ifri_mini_ml_lib.clustering.kmeans import KMeans

@pytest.fixture
def sample_data():
    data = np.array([
    [1, 2], [1, 3], [2, 2],      
    [8, 8], [8, 9], [9, 8],    
    [20, 20]  # Point éloigné pour mieux forcer une séparation
    ])
    return data


def test_init():
    kmeans = KMeans(n_clusters=2, max_iter=300, tol=1e-4, init=np.array([[1, 2], [8, 8]]), random_state=42 )
    assert kmeans.n_clusters == 2
    assert kmeans.max_iter == 300
    assert kmeans.labels is None
    print("Centroïdes initiaux :", kmeans.centroids)


def test_fit_predict(sample_data):
    kmeans = KMeans(n_clusters=3, max_iter=300, tol=1e-4, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(sample_data)

    # Vérifier que nous avons exactement 3 clusters
    assert len(set(labels)) == 3  

    # Les trois premiers points doivent être dans le même cluster
    assert len(set(labels[:3])) == 1  

    # Les trois suivants doivent être dans un autre cluster
    assert len(set(labels[3:6])) == 1  

    # Les deux groupes doivent être différents
    assert set(labels[:3]).isdisjoint(set(labels[3:6]))  

    # Vérifier que le dernier point est bien isolé dans un troisième cluster
    assert labels[-1] not in set(labels[:6])



def test_centroids_shape(sample_data):
    kmeans = KMeans(n_clusters=2, max_iter=100 )
    kmeans.fit_predict(sample_data)
    assert kmeans.centroids.shape == (2, sample_data.shape[1])

def test_predict_after_fit(sample_data):
    kmeans = KMeans(n_clusters=3, max_iter=100 )
    kmeans.fit(sample_data)
    # Prédire pour un nouveau point proche du cluster 0
    pred = kmeans.predict(np.array([[1.5, 2.5]]))
    assert pred[0] in {0, 1}

def test_plot_clusters_runs_without_error(sample_data):
    kmeans = KMeans(n_clusters=3, max_iter=100 )
    kmeans.fit_predict(sample_data)
    # Juste vérifier que la fonction plot_clusters s'exécute sans erreur
    kmeans.plot_clusters(sample_data)



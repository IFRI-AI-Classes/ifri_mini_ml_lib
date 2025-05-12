import numpy as np
import pytest
from ifri_mini_ml_lib.preprocessing.dimensionality_reduction import PCA

@pytest.fixture
def sample_data():
    return np.array([[2.5, 2.4],
                     [0.5, 0.7],
                     [2.2, 2.9],
                     [1.9, 2.2],
                     [3.1, 3.0],
                     [2.3, 2.7],
                     [2, 1.6],
                     [1, 1.1],
                     [1.5, 1.6],
                     [1.1, 0.9]])

def test_fit_sets_attributes(sample_data):
    pca = PCA(n_component=1).fit(sample_data)
    assert pca.mean is not None
    assert pca.cov.shape == (2, 2)  # Vérifie la forme de la matrice de covariance
    assert pca.eigen_values is not None
    assert pca.eigen_vectors is not None
    assert pca.components.shape == (1, 2)  # Correction ici (n_component, n_features)

def test_transform_output_shape(sample_data):
    pca = PCA(n_component=1).fit(sample_data)
    transformed = pca.transform(sample_data)
    assert transformed.shape == (10, 1)

def test_fit_transform_equivalence(sample_data):
    pca1 = PCA(n_component=1)
    transformed1 = pca1.fit_transform(sample_data)

    pca2 = PCA(n_component=1)
    pca2.fit(sample_data)
    transformed2 = pca2.transform(sample_data)

    assert np.allclose(transformed1, transformed2)

def test_explained_variance_ratio_sum(sample_data):
    pca = PCA(n_component=2).fit(sample_data)
    ratio = pca.explained_variances_ratio()
    assert np.allclose(np.sum(ratio), 1.0, atol=1e-6)

def test_explained_variances_sorted(sample_data):
    pca = PCA(n_component=2).fit(sample_data)
    eigenvalues = pca.explained_variances()
    assert np.all(np.diff(eigenvalues) <= 0)  # Vérifie l'ordre décroissant

def test_mean_calculation(sample_data):
    pca = PCA(n_component=1).fit(sample_data)
    expected_mean = np.mean(sample_data, axis=0)
    assert np.allclose(pca.mean, expected_mean)

def test_cov_matrix_shape(sample_data):
    pca = PCA(n_component=1).fit(sample_data)
    assert pca.cov.shape == (2, 2)  # Pour 2 variables, matrice 2x2
import pytest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE as skTSNE


# Votre classe TSNE doit être importée ici
from ifri_mini_ml_lib.preprocessing.dimensionality_reduction import TSNE

@pytest.fixture
def test_data():
    """Génère des données de test pour tous les tests"""
    X, y = make_blobs(n_samples=100, centers=3, n_features=5, random_state=42)
    return X, y

def test_initialization():
    """Teste l'initialisation correcte des paramètres"""
    tsne = TSNE(n_components=2, perplexity=20, learning_rate=300)
    
    assert tsne.n_components == 2
    assert tsne.perplexity == 20
    assert tsne.learning_rate == 300
    assert tsne.n_iter == 1000

def test_input_validation(test_data):
    """Teste la validation des entrées invalides"""
    X, _ = test_data
    tsne = TSNE(perplexity=50)  # 3*50 = 150 > 100 samples
    
    with pytest.raises(ValueError) as excinfo:
        tsne.fit(X)
    assert "must be at least 3 * perplexity" in str(excinfo.value)

def test_output_shape(test_data):
    """Vérifie la forme des sorties"""
    X, _ = test_data
    tsne = TSNE(n_components=3, n_iter=100)
    
    embedding = tsne.fit_transform(X)
    
    assert embedding.shape == (X.shape[0], 3)
    assert tsne.embedding_.shape == (X.shape[0], 3)
    assert isinstance(tsne.kl_divergence_, float)
    assert tsne.n_iter_ <= tsne.n_iter

def test_determinism(test_data):
    """Teste la reproductibilité avec random_state"""
    X, _ = test_data
    
    tsne1 = TSNE(random_state=42, n_iter=150)
    emb1 = tsne1.fit_transform(X)
    
    tsne2 = TSNE(random_state=42, n_iter=150)
    emb2 = tsne2.fit_transform(X)
    
    np.testing.assert_allclose(emb1, emb2, atol=1e-5)

def test_kl_divergence(test_data):
    """Vérifie que la KL divergence est calculée et cohérente"""
    X, _ = test_data
    tsne = TSNE(n_iter=200)
    tsne.fit(X)
    
    assert tsne.kl_divergence_ > 0
    assert not np.isnan(tsne.kl_divergence_)
    assert isinstance(tsne.kl_divergence_, float)

def test_early_exaggeration_effect(test_data):
    """Teste l'effet de l'early exaggeration"""
    X, _ = test_data
    
    tsne1 = TSNE(early_exaggeration=12.0, n_iter=300)
    tsne1.fit(X)
    
    tsne2 = TSNE(early_exaggeration=1.0, n_iter=300)
    tsne2.fit(X)
    
    # Vérifie que les résultats sont significativement différents
    assert not np.allclose(tsne1.embedding_, tsne2.embedding_, atol=0.1)

def test_compare_with_sklearn_basic(test_data):
    """Comparaison de base avec l'implémentation sklearn"""
    X, y = test_data
    
    # Votre implémentation
    custom_tsne = TSNE(n_components=2, random_state=42, n_iter=500)
    custom_emb = custom_tsne.fit_transform(X)
    
    # Implémentation sklearn
    sklearn_tsne = skTSNE(n_components=2, random_state=42, n_iter=500)
    sklearn_emb = sklearn_tsne.fit_transform(X)
    
    # Vérifications de base
    assert custom_emb.shape == sklearn_emb.shape
    
    # Les variances doivent être du même ordre de grandeur
    assert np.var(custom_emb) == pytest.approx(np.var(sklearn_emb), rel=0.5)
    
    # Vérifie que les clusters sont séparés (pour les données de blobs)
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    custom_score = adjusted_rand_score(y, custom_emb.argmax(axis=1))
    sklearn_score = adjusted_rand_score(y, sklearn_emb.argmax(axis=1))
    assert abs(custom_score - sklearn_score) < 0.2
    """
    custom_score = silhouette_score(custom_emb, y)
    sklearn_score = silhouette_score(sklearn_emb, y)
    
    assert custom_score > 0.6  # Doit avoir une bonne séparation
    assert abs(custom_score - sklearn_score) < 0.2 """


def test_gradient_computation():
    """Test avec des entrées non-symétriques réalistes"""
    Y = np.array([[0.1, 0.2], [1.1, 0.8], [-0.5, -0.3]])
    P = np.array([[0, 0.8, 0.2], [0.8, 0, 0.2], [0.2, 0.2, 0]])
    Q = np.array([[0, 0.1, 0.9], [0.1, 0, 0.9], [0.9, 0.1, 0]])

    tsne = TSNE(n_components=2)
    gradient = tsne._compute_gradient(P, Q, Y)

    assert gradient.shape == (3, 2)
    assert not np.allclose(gradient, 0, atol=1e-3)

def test_probability_calculations():
    """Teste les calculs de probabilités P et Q"""
    X = np.random.randn(10, 5)
    tsne = TSNE(perplexity=5)
    
    P = tsne._compute_joint_probabilities(X, 5)
    Q = tsne._compute_low_dimensional_probabilities(np.random.randn(10, 2))
    
    # Vérifications de base
    assert P.shape == (10, 10)
    assert Q.shape == (10, 10)
    assert np.all(P >= 0)
    assert np.all(Q >= 0)
    assert pytest.approx(P.sum()) == 1.0
    assert pytest.approx(Q.sum()) == 1.0
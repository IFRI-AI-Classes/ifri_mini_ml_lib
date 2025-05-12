import pytest
import numpy as np

# Votre classe TSNE doit être importée ici
from ifri_mini_ml_lib.preprocessing.dimensionality_reduction import TSNE

@pytest.fixture
def test_data():
    """Génère des données de test manuellement"""
    np.random.seed(42)
    
    # Création de 3 clusters manuellement
    cluster1 = np.random.normal(loc=5.0, scale=1.0, size=(34, 5))
    cluster2 = np.random.normal(loc=0.0, scale=1.0, size=(33, 5))
    cluster3 = np.random.normal(loc=-5.0, scale=1.0, size=(33, 5))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    np.random.shuffle(X)
    
    # Création d'étiquettes factices (non utilisées dans les tests)
    y = np.array([0]*34 + [1]*33 + [2]*33)
    
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
    X = np.random.normal(size=(10, 5))
    tsne = TSNE(perplexity=5)
    
    P = tsne._compute_joint_probabilities(X, 5)
    Q = tsne._compute_low_dimensional_probabilities(np.random.normal(size=(10, 2)))
    
    # Vérifications de base
    assert P.shape == (10, 10)
    assert Q.shape == (10, 10)
    assert np.all(P >= 0)
    assert np.all(Q >= 0)
    assert pytest.approx(P.sum()) == 1.0
    assert pytest.approx(Q.sum()) == 1.0

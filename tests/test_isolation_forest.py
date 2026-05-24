"""Unit tests for Isolation Forest Anomaly Detection"""
import sys
import os
import pytest
import numpy as np
 
# ============================================================================
# IMPORT MANAGEMENT (compatible with project structure)
# ============================================================================
 
def import_isolation_forest():
    """Attempts to import IsolationForest from different possible paths"""
 
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'ifri_mini_ml_lib', 'anomalies_detection'),
        os.path.join(os.path.dirname(__file__), '..', 'ifri_ml_mini', 'anomalies_detection'),
        os.path.join(os.path.dirname(__file__), '..'),
        os.path.dirname(__file__),
    ]
 
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            try:
                from isolation_forest import IsolationForest
                print(f"Import successful from: {path}")
                return IsolationForest
            except ImportError:
                continue
 
    raise ImportError("Unable to import IsolationForest. Please check the paths.")
 
IsolationForest = import_isolation_forest()
 
 
# =================================================
# FIXTURES
# =================================================
 
@pytest.fixture
def normal_data():
    """200 points normaux groupés autour de l'origine."""
    np.random.seed(42)
    return np.random.randn(200, 2)
 
 
@pytest.fixture
def data_with_anomalies():
    """200 points normaux + 3 anomalies évidentes très éloignées."""
    np.random.seed(42)
    X_normal = np.random.randn(200, 2)
    X_anomalies = np.array([[10, 10], [-10, -10], [8, -8]])
    return np.vstack([X_normal, X_anomalies])
 
 
@pytest.fixture
def fitted_model(data_with_anomalies):
    """Modèle déjà entraîné — évite de répéter fit() dans chaque test."""
    model = IsolationForest(n_trees=100, sample_size=256, contamination=0.05)
    model.fit(data_with_anomalies)
    return model, data_with_anomalies
 
 
# ==========================================
# TESTS D'INITIALISATION
# ==========================================
 
class TestInit:
    """Vérifie que le modèle est bien configuré à la création."""
 
    def test_default_parameters(self):
        """Les valeurs par défaut doivent correspondre à celles de l'implémentation."""
        model = IsolationForest()
        assert model.n_trees == 100
        assert model.sample_size == 256
        assert model.contamination == 0.1
        assert model.trees == []
 
    def test_custom_parameters(self):
        """Les paramètres passés explicitement doivent être enregistrés tels quels."""
        model = IsolationForest(n_trees=50, sample_size=128, contamination=0.05)
        assert model.n_trees == 50
        assert model.sample_size == 128
        assert model.contamination == 0.05
 
 
# ===================================
# TESTS DE FIT
# ===================================
 
class TestFit:
    """Vérifie que fit() construit correctement la forêt."""
 
    def test_fit_returns_self(self, normal_data):
        """fit() doit retourner l'instance elle-même pour permettre le chaînage."""
        # Exemple de chaînage : model.fit(X).predict(X)
        model = IsolationForest()
        result = model.fit(normal_data)
        assert result is model
 
    def test_fit_builds_correct_number_of_trees(self, normal_data):
        """fit() doit construire exactement n_trees arbres, ni plus ni moins."""
        model = IsolationForest(n_trees=50)
        model.fit(normal_data)
        assert len(model.trees) == 50
 
    def test_fit_computes_threshold(self, normal_data):
        """fit() doit calculer un seuil de décision qui est un float valide."""
        model = IsolationForest()
        model.fit(normal_data)
        assert hasattr(model, 'threshold')
        assert isinstance(float(model.threshold), float)
 
    def test_fit_accepts_various_input_types(self):
        """fit() doit accepter aussi bien une liste Python qu'un tableau numpy."""
        # Liste Python
        model_list = IsolationForest(n_trees=10)
        model_list.fit([[1, 2], [3, 4], [5, 6]] * 40)
        assert len(model_list.trees) == 10
 
        # Tableau numpy
        model_np = IsolationForest(n_trees=10)
        model_np.fit(np.random.randn(100, 2))
        assert len(model_np.trees) == 10
 
    def test_fit_sample_size_capped_to_n_samples(self):
        """Si n < sample_size, fit() ne doit pas planter (il prend tous les points)."""
        # 50 points seulement mais sample_size=256 par défaut
        X = np.random.randn(50, 2)
        model = IsolationForest(n_trees=10, sample_size=256)
        model.fit(X)
        assert len(model.trees) == 10
 
 
# ===================================
# TESTS D'ANOMALY SCORE
# ===================================
 
class TestAnomalyScore:
    """Vérifie que les scores produits ont la bonne forme et le bon sens."""
 
    def test_scores_shape(self, fitted_model):
        """anomaly_score() doit retourner un tableau de longueur n_samples."""
        model, X = fitted_model
        scores = model.anomaly_score(X)
        assert scores.shape == (len(X),)
 
    def test_scores_between_0_and_1(self, fitted_model):
        """Tous les scores doivent être dans [0, 1] — c'est la définition du score IF."""
        model, X = fitted_model
        scores = model.anomaly_score(X)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
 
    def test_anomalies_score_higher_than_normal(self, fitted_model):
        """Les 3 anomalies connues doivent avoir un score moyen supérieur aux normaux.
 
        C'est le test fonctionnel central : si l'algo marche, les points isolés
        (loin du nuage) ont des chemins courts → score élevé.
        """
        model, X = fitted_model
        scores = model.anomaly_score(X)
 
        anomaly_scores = scores[-3:]      # les 3 derniers points = anomalies
        normal_mean    = scores[:200].mean()
 
        assert anomaly_scores.mean() > normal_mean
 
    def test_obvious_anomaly_has_high_score(self, fitted_model):
        """Un point très éloigné du nuage doit avoir un score supérieur à 0.6."""
        model, _ = fitted_model
        obvious_anomaly = np.array([[10, 10]])
        score = model.anomaly_score(obvious_anomaly)
        assert score[0] > 0.6
 
 
# ====================================
# TESTS DE PREDICT
# ====================================
 
class TestPredict:
    """Vérifie que predict() produit des labels cohérents."""
 
    def test_predict_output_format(self, fitted_model):
        """predict() doit retourner un tableau binaire (0/1) de taille n_samples."""
        model, X = fitted_model
        labels = model.predict(X)
 
        assert labels.shape == (len(X),)
        assert set(labels).issubset({0, 1})
 
    def test_predict_detects_obvious_anomalies(self, fitted_model):
        """Les 3 anomalies évidentes doivent toutes être labellisées 1."""
        model, X = fitted_model
        labels = model.predict(X)
 
        assert labels[-1] == 1
        assert labels[-2] == 1
        assert labels[-3] == 1
 
    def test_contamination_rate_respected(self, data_with_anomalies):
        """Le taux de points détectés doit être très proche du taux de contamination.
 
        Par construction, fit() calcule le threshold comme le percentile
        (1 - contamination), donc predict() doit retourner exactement
        contamination * n points comme anomalies.
        """
        contamination = 0.05
        model = IsolationForest(n_trees=100, contamination=contamination)
        model.fit(data_with_anomalies)
        labels = model.predict(data_with_anomalies)
 
        detected_rate = labels.sum() / len(labels)
        assert abs(detected_rate - contamination) < 0.03
 
    def test_predict_on_unseen_data(self, normal_data):
        """predict() doit fonctionner sur de nouvelles données non vues pendant fit().
 
        Ce test vérifie que le modèle généralise bien :
        [15, 15] est une anomalie évidente même si le modèle n'a jamais vu ce point.
        """
        model = IsolationForest(n_trees=50)
        model.fit(normal_data)
 
        X_new = np.array([[0, 0], [0.5, -0.5], [15, 15]])
        labels = model.predict(X_new)
 
        assert labels.shape == (3,)
        assert labels[-1] == 1  # [15, 15] doit être détecté
 
 
# ====================================
# TESTS DE ROBUSTESSE
# ====================================
 
class TestRobustness:
    """Vérifie que le modèle tient sur des cas limites ou inhabituels."""
 
    def test_works_with_1d_data(self):
        """L'algo doit fonctionner avec des données à une seule dimension."""
        X = np.vstack([np.random.randn(100, 1), [[10]]])
        model = IsolationForest(n_trees=50)
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (101,)
 
    def test_works_with_high_dimensional_data(self):
        """L'algo doit fonctionner avec des données à haute dimension (20 features)."""
        X = np.random.randn(200, 20)
        model = IsolationForest(n_trees=50)
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (200,)
 
    def test_reproducibility_with_seed(self, data_with_anomalies):
        """Deux modèles entraînés avec la même graine doivent produire les mêmes scores.
 
        Sans cette propriété, les tests fonctionnels deviendraient non déterministes
        et on ne pourrait pas déboguer facilement.
        """
        np.random.seed(0)
        model1 = IsolationForest(n_trees=50, sample_size=128)
        model1.fit(data_with_anomalies)
        scores1 = model1.anomaly_score(data_with_anomalies)
 
        np.random.seed(0)
        model2 = IsolationForest(n_trees=50, sample_size=128)
        model2.fit(data_with_anomalies)
        scores2 = model2.anomaly_score(data_with_anomalies)
 
        np.testing.assert_array_almost_equal(scores1, scores2)
 
    def test_normal_data_produces_few_anomalies(self, normal_data):
        """Sur des données vraiment normales, le taux de faux positifs doit rester bas."""
        model = IsolationForest(n_trees=100, contamination=0.05)
        model.fit(normal_data)
        labels = model.predict(normal_data)
 
        anomaly_rate = labels.sum() / len(labels)
        assert anomaly_rate < 0.10
 
 
# ====================================
# DIRECT EXECUTION
# ====================================
 
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING ISOLATION FOREST TESTS")
    print("=" * 60)
 
    np.random.seed(42)
    X_normal = np.random.randn(200, 2)
    X_anomalies = np.array([[10, 10], [-10, -10], [8, -8]])
    X = np.vstack([X_normal, X_anomalies])
 
    model = IsolationForest(n_trees=100, sample_size=256, contamination=0.05)
    model.fit(X)
 
    labels = model.predict(X)
    scores = model.anomaly_score(X)
 
    print(f"Total points      : {len(X)}")
    print(f"Anomalies trouvées: {labels.sum()}")
    print()
    print("=== Scores des anomalies connues ===")
    print(f"Point [10,  10]  → score: {scores[-3]:.4f} | label: {labels[-3]}")
    print(f"Point [-10, -10] → score: {scores[-2]:.4f} | label: {labels[-2]}")
    print(f"Point [8,  -8]   → score: {scores[-1]:.4f} | label: {labels[-1]}")
 
    if labels[-1] == 1 and labels[-2] == 1 and labels[-3] == 1:
        print("\nBASIC TEST PASSED!")
    else:
        print("\nBASIC TEST FAILED")
 
    print("\n" + "=" * 60)
    print("To run all tests: pytest tests/test_isolation_forest.py -v")
    print("=" * 60)
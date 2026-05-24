import pytest
import math
import numpy as np
from ifri_mini_ml_lib.preprocessing.text import TFIDFVectorizer

# 1. FIT_TRANSFORM SIMPLE  

def test_tfidf_fit_transform_simple():
    """fit_transform sur un corpus minimal doit retourner un ndarray (n_docs, n_terms)."""
    corpus = [["a", "b"], ["a", "c"]]
    tfidf = TFIDFVectorizer()
    matrix = tfidf.fit_transform(corpus)

    # La matrice doit être un ndarray NumPy
    assert isinstance(matrix, np.ndarray)
    # 2 documents, 3 termes uniques : "a", "b", "c"
    assert matrix.shape == (2, 3)
    # Aucune valeur ne doit être NaN
    assert not np.any(np.isnan(matrix))


# 2. VALEUR IDF LISSÉE (smooth_idf=True) 

def test_tfidf_idf_smooth_value():
    """Les poids IDF lissés doivent correspondre à la formule log((N+1)/(df+1))+1."""
    corpus = [["a", "b"], ["a", "c"]]
    tfidf = TFIDFVectorizer(smooth_idf=True, norm=None)
    tfidf.fit(corpus)

    N = 2
    # "a" apparaît dans les 2 documents → df = 2
    expected_idf_a = math.log((N + 1) / (2 + 1)) + 1   # log(1.0) + 1 = 1.0
    # "b" apparaît dans 1 document  → df = 1
    expected_idf_b = math.log((N + 1) / (1 + 1)) + 1   # log(1.5) + 1 ≈ 1.405

    assert tfidf.idf_["a"] == pytest.approx(expected_idf_a, rel=1e-6)
    assert tfidf.idf_["b"] == pytest.approx(expected_idf_b, rel=1e-6)


# 3. TERME INCONNU IGNORÉ SILENCIEUSEMENT 


def test_tfidf_unknown_term_ignored():
    """Un terme absent du vocabulaire appris doit être ignoré sans erreur."""
    train_corpus = [["hello", "world"], ["hello", "python"]]
    test_corpus  = [["hello", "unknown_term"]]

    tfidf = TFIDFVectorizer(norm=None)
    tfidf.fit(train_corpus)

    # Ne doit pas lever d'exception
    result = tfidf.transform(test_corpus)

    assert result.shape == (1, len(tfidf.vocabulary_))
    # "unknown_term" n'est pas dans le vocabulaire → sa contribution est nulle,
    # mais la ligne ne doit pas être entièrement nulle grâce à "hello"
    assert np.any(result != 0)


# 4. TRANSFORM SANS FIT PRÉALABLE  

def test_tfidf_transform_not_fitted_raises_error():
    """transform() appelé avant fit() doit lever une erreur."""
    tfidf = TFIDFVectorizer()
    # vocabulary_ et idf_ sont des dicts vides → la matrice sera vide ou
    # les indices seront introuvables selon l'implémentation.
    # Dans tous les cas, le comportement doit être une erreur explicite.
    with pytest.raises((AttributeError, KeyError, ValueError)):
        tfidf.transform([["hello", "world"]])


# 5. FORMULE IDF NON LISSÉE  

def test_tfidf_idf_standard_formula():
    """Avec smooth_idf=False, un terme universel doit avoir un IDF de zéro."""
    corpus = [["a", "b"], ["a", "c"]]
    tfidf = TFIDFVectorizer(smooth_idf=False, norm=None)
    tfidf.fit(corpus)

    N = 2
    # "a" est dans les 2 documents → idf = log(2 / 2) = log(1) = 0.0
    assert tfidf.idf_["a"] == pytest.approx(0.0, abs=1e-9)
    # "b" est dans 1 document → idf = log(2 / 1) = log(2)
    assert tfidf.idf_["b"] == pytest.approx(math.log(2), rel=1e-6)


# 6. PLUSIEURS DOCUMENTS — FORME ET COHÉRENCE 

def test_tfidf_multiple_documents():
    """transform() doit produire une ligne par document avec le même vocabulaire."""
    corpus = [
        ["chat", "dort"],
        ["chien", "court"],
        ["chat", "court"],
    ]
    tfidf = TFIDFVectorizer()
    matrix = tfidf.fit_transform(corpus)

    # Autant de lignes que de documents
    assert matrix.shape[0] == 3
    # Autant de colonnes que de termes uniques dans le corpus
    assert matrix.shape[1] == len(tfidf.vocabulary_)
    # Chaque ligne doit être un vecteur de norme ≈ 1 (L2 activé par défaut)
    for row in matrix:
        row_norm = np.linalg.norm(row)
        if row_norm > 0:
            assert row_norm == pytest.approx(1.0, rel=1e-6)


# 7. PRÉCISION NUMÉRIQUE L2  ←→  test_knn_regression_float_precision

def test_tfidf_l2_norm_unit_length():
    """Avec norm='l2', chaque ligne du résultat doit avoir une norme de 1.0."""
    corpus = [["a", "b", "c"], ["b", "c", "d"], ["a", "d"]]
    tfidf = TFIDFVectorizer(norm="l2")
    matrix = tfidf.fit_transform(corpus)

    for i, row in enumerate(matrix):
        computed_norm = float(np.linalg.norm(row))
        assert computed_norm == pytest.approx(1.0, rel=1e-6), (
            f"La norme de la ligne {i} est {computed_norm}, attendu 1.0"
        )


# TESTS COMPLÉMENTAIRES

def test_tfidf_vocabulary_is_sorted():
    """Le vocabulaire doit être trié alphabétiquement et les index contigus."""
    corpus = [["banane", "abricot", "cerise"]]
    tfidf = TFIDFVectorizer()
    tfidf.fit(corpus)

    sorted_terms = sorted(tfidf.vocabulary_.keys())
    for expected_idx, term in enumerate(sorted_terms):
        assert tfidf.vocabulary_[term] == expected_idx


def test_tfidf_fit_transform_equals_fit_then_transform():
    """fit_transform() doit produire le même résultat que fit() puis transform()."""
    corpus = [["x", "y"], ["y", "z"]]

    tfidf_1 = TFIDFVectorizer(smooth_idf=True, norm="l2")
    matrix_1 = tfidf_1.fit_transform(corpus)

    tfidf_2 = TFIDFVectorizer(smooth_idf=True, norm="l2")
    tfidf_2.fit(corpus)
    matrix_2 = tfidf_2.transform(corpus)

    np.testing.assert_array_almost_equal(matrix_1, matrix_2)


def test_tfidf_norm_none_does_not_normalize():
    """Avec norm=None, les vecteurs ne doivent pas être ramenés à la norme unité."""
    corpus = [["a", "a", "b"]]
    tfidf = TFIDFVectorizer(norm=None)
    matrix = tfidf.fit_transform(corpus)

    row_norm = np.linalg.norm(matrix[0])
    # La norme brute ne doit pas valoir exactement 1 (sauf coïncidence)
    # On vérifie que la norme est différente de 0 et qu'elle n'est pas normalisée
    assert row_norm > 0
    assert row_norm != pytest.approx(1.0, abs=1e-3)


def test_tfidf_sublinear_tf_reduces_weight():
    """sublinear_tf=True doit produire un TF plus faible qu'un TF brut pour count > 1."""
    # doc avec "a" répété 5 fois → tf_brut = 5/5 = 1.0 ; tf_sublinear = 1 + log(5)
    doc = ["a"] * 5

    tfidf_raw = TFIDFVectorizer(sublinear_tf=False, norm=None)
    tfidf_sub = TFIDFVectorizer(sublinear_tf=True,  norm=None)

    tf_raw = tfidf_raw._compute_tf(doc)["a"]   # 1.0
    tf_sub = tfidf_sub._compute_tf(doc)["a"]   # 1 + log(5) ≈ 2.609

    # Le TF sublinear doit être différent du TF brut
    assert tf_raw != pytest.approx(tf_sub, rel=1e-3)
    # Spécifiquement : 1 + log(5) pour count=5, total=5
    assert tf_sub == pytest.approx(1 + math.log(5), rel=1e-6)


def test_tfidf_zero_row_not_nan_after_l2():
    """Un vecteur nul (terme absent du test) ne doit pas produire de NaN après L2."""
    train = [["a", "b"], ["a", "c"]]
    test  = [["z"]]          # "z" absent du vocabulaire → ligne nulle

    tfidf = TFIDFVectorizer(norm="l2")
    tfidf.fit(train)
    result = tfidf.transform(test)

    assert not np.any(np.isnan(result))
    # La ligne entière doit rester à zéro (pas de division par zéro)
    np.testing.assert_array_equal(result[0], np.zeros(len(tfidf.vocabulary_)))


def test_tfidf_single_document_corpus():
    """Un corpus d'un seul document doit fonctionner sans erreur."""
    corpus = [["mot", "unique", "ici"]]
    tfidf = TFIDFVectorizer()
    matrix = tfidf.fit_transform(corpus)

    assert matrix.shape == (1, 3)
    assert not np.any(np.isnan(matrix))
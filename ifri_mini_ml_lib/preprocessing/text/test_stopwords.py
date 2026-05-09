import pytest
from ifri_mini_ml_lib.preprocessing.text.stop_word import StopWordRemover


@pytest.fixture
def remover_en():
    return StopWordRemover(language="english")


def test_suppression_basique_anglais(remover_en):
    """Teste la suppression de stop words en anglais"""
    tokens = ["the", "cat", "is", "on", "the", "roof"]
    result = remover_en.transform(tokens)
    assert "the" not in result
    assert "is" not in result
    assert "cat" in result
    assert "roof" in result


def test_liste_de_documents(remover_en):
    """Teste le traitement de plusieurs documents à la fois"""
    docs = [
        ["the", "film", "is", "great"],
        ["a", "bad", "movie", "but", "interesting"]
    ]
    result = remover_en.transform(docs)
    assert "the" not in result[0]
    assert "film" in result[0]
    assert "a" not in result[1]
    assert "interesting" in result[1]


def test_input_vide(remover_en):
    """Teste le comportement avec une liste vide"""
    assert remover_en.transform([]) == []


def test_casse_insensible(remover_en):
    """Teste que la suppression est insensible à la casse"""
    tokens = ["The", "THE", "cat", "IS"]
    result = remover_en.transform(tokens)
    assert "The" not in result
    assert "THE" not in result
    assert "IS" not in result
    assert "cat" in result


def test_fit_retourne_self(remover_en):
    """Teste que fit() retourne bien self"""
    result = remover_en.fit()
    assert result is remover_en


def test_get_stopwords(remover_en):
    """Teste que get_stopwords retourne bien un set non vide"""
    sw = remover_en.get_stopwords()
    assert isinstance(sw, set)
    assert len(sw) > 0
    assert "the" in sw


def test_langue_inconnue():
    """Langue non supportée = aucun mot supprimé"""
    remover = StopWordRemover(language="arabic")
    tokens = ["le", "film", "est", "bon"]
    assert remover.transform(tokens) == tokens



@pytest.fixture
def remover_fr():
    return StopWordRemover(language="french")


def test_suppression_basique_francais(remover_fr):
    """Teste la suppression de stop words en français"""
    tokens = ["le", "film", "est", "vraiment", "excellent"]
    result = remover_fr.transform(tokens)
    assert "le" not in result
    assert "est" not in result
    assert "film" in result
    assert "excellent" in result


def test_custom_stopwords():
    """Teste l'ajout de stop words personnalisés"""
    remover = StopWordRemover(language="english", custom_stopwords=["film", "movie"])
    tokens = ["the", "film", "is", "great"]
    result = remover.transform(tokens)
    assert "film" not in result
    assert "great" in result

    
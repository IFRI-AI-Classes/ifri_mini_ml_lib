import pytest
from ifri_mini_ml_lib.preprocessing.preparation.tokenization import Tokenizer


# 1. TOKENISATION DE BASE

def test_tokenizer_simple_sentence():
    """Pipeline par défaut sur une phrase sans cas particuliers."""
    tokenizer = Tokenizer()
    result = tokenizer.tokenize("Hello world")
    assert result == ["hello", "world"]


def test_tokenizer_lowercase_applied():
    """Les majuscules doivent être converties en minuscules."""
    tokenizer = Tokenizer()
    result = tokenizer.tokenize("HELLO World")
    assert result == ["hello", "world"]


def test_tokenizer_lowercase_disabled():
    """Quand lowercase=False, la casse originale doit être préservée."""
    tokenizer = Tokenizer(lowercase=False)
    result = tokenizer.tokenize("HELLO World")
    assert result == ["HELLO", "World"]


# 2. GESTION DES CONTRACTIONS

def test_tokenizer_contraction_negative():
    """Les contractions négatives doivent être développées correctement."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize("I can't go")      == ["i", "can", "not", "go"]
    assert tokenizer.tokenize("she won't come")  == ["she", "won't", "come".replace("won't", "will not")]


def test_tokenizer_contraction_auxiliary():
    """Les suffixes auxiliaires ('re, 'm, 'll, 'd, 've) doivent être développés."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize("they're happy")   == ["they", "are", "happy"]
    assert tokenizer.tokenize("I'm tired")       == ["i", "am", "tired"]
    assert tokenizer.tokenize("she'll come")     == ["she", "will", "come"]
    assert tokenizer.tokenize("I'd like that")   == ["i", "would", "like", "that"]
    assert tokenizer.tokenize("we've done it")   == ["we", "have", "done", "it"]


def test_tokenizer_contraction_pronoun_is():
    """Pronom + 's doit être interprété comme 'pronom + is'."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize("it's great")      == ["it", "is", "great"]
    assert tokenizer.tokenize("he's ready")      == ["he", "is", "ready"]
    assert tokenizer.tokenize("she's here")      == ["she", "is", "here"]


def test_tokenizer_contraction_possessive_stripped():
    """Le 's possessif (non pronominal) doit être supprimé."""
    tokenizer = Tokenizer()
    result = tokenizer.tokenize("John's book")
    # "John's" → "John" (le 's est retiré) puis lowercase → "john"
    assert "john" in result
    assert "book" in result
    assert "s" not in result


def test_tokenizer_contractions_disabled():
    """Quand handle_contractions=False, les contractions restent intactes."""
    tokenizer = Tokenizer(
        handle_contractions=False,
        handle_apostrophes=False,
        remove_punctuation=False,
        lowercase=False,
    )
    result = tokenizer.tokenize("I can't")
    assert "can't" in result


# 3. GESTION DES APOSTROPHES

def test_tokenizer_apostrophe_ascii():
    """L'apostrophe ASCII (') doit être remplacée par un espace."""
    tokenizer = Tokenizer(handle_contractions=False)
    result = tokenizer.tokenize("l'homme")
    assert "l" in result
    assert "homme" in result


def test_tokenizer_apostrophe_unicode():
    """L'apostrophe Unicode U+2019 doit être remplacée par un espace."""
    tokenizer = Tokenizer(handle_contractions=False)
    result = tokenizer.tokenize("l\u2019homme")
    assert "l" in result
    assert "homme" in result


def test_tokenizer_apostrophe_disabled():
    """Quand handle_apostrophes=False, les apostrophes ne sont pas touchées."""
    tokenizer = Tokenizer(
        handle_contractions=False,
        handle_apostrophes=False,
        remove_punctuation=False,
        lowercase=False,
    )
    result = tokenizer.tokenize("l'homme")
    assert result == ["l'homme"]


# 4. SUPPRESSION DE LA PONCTUATION

def test_tokenizer_punctuation_removed():
    """Les signes de ponctuation courants doivent être éliminés."""
    tokenizer = Tokenizer()
    result = tokenizer.tokenize("Hello, world! How are you?")
    assert result == ["hello", "world", "how", "are", "you"]


def test_tokenizer_punctuation_preserves_accented():
    """Les caractères accentués (À–ÿ) ne doivent pas être supprimés."""
    tokenizer = Tokenizer()
    result = tokenizer.tokenize("élève naïf façade")
    assert "élève" in result
    assert "naïf" in result
    assert "façade" in result


def test_tokenizer_punctuation_preserves_digits():
    """Les chiffres ne doivent pas être supprimés par remove_punctuation."""
    tokenizer = Tokenizer()
    result = tokenizer.tokenize("room 404 error")
    assert "404" in result


def test_tokenizer_punctuation_disabled():
    """Quand remove_punctuation=False, la ponctuation reste dans les tokens."""
    tokenizer = Tokenizer(
        lowercase=False,
        handle_contractions=False,
        handle_apostrophes=False,
        remove_punctuation=False,
    )
    result = tokenizer.tokenize("Hello, world!")
    assert "Hello," in result
    assert "world!" in result


# 5. FILTRE PAR LONGUEUR MINIMALE

def test_tokenizer_min_token_length_filters_short():
    """Les tokens plus courts que min_token_length doivent être écartés."""
    tokenizer = Tokenizer(min_token_length=3)
    result = tokenizer.tokenize("I am a student")
    # "i", "am", "a" font moins de 3 caractères → retirés
    assert result == ["student"]


def test_tokenizer_min_token_length_zero_keeps_all():
    """Avec min_token_length=0, aucun token ne doit être retiré."""
    tokenizer = Tokenizer(min_token_length=0, remove_punctuation=False, lowercase=False)
    result = tokenizer.tokenize("I am")
    assert result == ["I", "am"]


def test_tokenizer_min_token_length_exact_boundary():
    """Un token dont la longueur est exactement min_token_length doit être conservé."""
    tokenizer = Tokenizer(min_token_length=2)
    result = tokenizer.tokenize("go do it now")
    # "go"(2) → gardé ; "do"(2) → gardé ; "it"(2) → gardé ; "now"(3) → gardé
    assert "go" in result
    assert "do" in result
    assert "it" in result


 
# 6. CAS LIMITES (EDGE CASES)

def test_tokenizer_empty_string():
    """Une chaîne vide doit retourner une liste vide."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize("") == []


def test_tokenizer_only_punctuation():
    """Une chaîne composée uniquement de ponctuation doit retourner []."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize("!!! ??? ...") == []


def test_tokenizer_only_spaces():
    """Une chaîne d'espaces doit retourner une liste vide."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize("     ") == []


def test_tokenizer_multiple_spaces_between_words():
    """Les espaces multiples entre mots ne doivent pas créer de tokens vides."""
    tokenizer = Tokenizer()
    result = tokenizer.tokenize("hello     world")
    assert result == ["hello", "world"]


def test_tokenizer_all_flags_disabled():
    """Avec tous les flags désactivés, le texte est uniquement splitté."""
    tokenizer = Tokenizer(
        lowercase=False,
        handle_contractions=False,
        handle_apostrophes=False,
        remove_punctuation=False,
        min_token_length=0,
    )
    result = tokenizer.tokenize("Hello World")
    assert result == ["Hello", "World"]


# 7. TOKENISATION DE CORPUS

def test_tokenizer_corpus_simple():
    """tokenize_corpus doit appliquer tokenize() à chaque document."""
    tokenizer = Tokenizer()
    corpus = ["Hello world", "I am fine"]
    result = tokenizer.tokenize_corpus(corpus)
    assert result == [["hello", "world"], ["i", "am", "fine"]]


def test_tokenizer_corpus_empty():
    """Un corpus vide doit retourner une liste vide."""
    tokenizer = Tokenizer()
    assert tokenizer.tokenize_corpus([]) == []


def test_tokenizer_corpus_single_document():
    """Un corpus à un seul document doit retourner une liste avec une sous-liste."""
    tokenizer = Tokenizer()
    result = tokenizer.tokenize_corpus(["good morning"])
    assert len(result) == 1
    assert result[0] == ["good", "morning"]


def test_tokenizer_corpus_preserves_order():
    """L'ordre des documents dans le corpus doit être préservé."""
    tokenizer = Tokenizer()
    corpus = ["first doc", "second doc", "third doc"]
    result = tokenizer.tokenize_corpus(corpus)
    assert result[0] == ["first", "doc"]
    assert result[1] == ["second", "doc"]
    assert result[2] == ["third", "doc"]


def test_tokenizer_corpus_length_matches_input():
    """La longueur de la sortie doit être égale à celle du corpus."""
    tokenizer = Tokenizer()
    corpus = ["a", "b", "c", "d", "e"]
    assert len(tokenizer.tokenize_corpus(corpus)) == len(corpus)


# 8. PIPELINE COMPLET (intégration)

def test_tokenizer_full_pipeline_contraction_and_punctuation():
    """Le pipeline complet doit traiter contractions ET ponctuation ensemble."""
    tokenizer = Tokenizer()
    result = tokenizer.tokenize("I can't believe it's over!")
    # "can't" → "can not" ; "it's" → "it is" ; "!" supprimé
    assert "can" in result
    assert "not" in result
    assert "it" in result
    assert "is" in result
    assert "over" in result


def test_tokenizer_full_pipeline_order_matters():
    """
    La contraction doit être traitée avant la suppression des apostrophes
    pour éviter que "can't" ne devienne "can t" au lieu de "can not".
    """
    tokenizer = Tokenizer()
    result = tokenizer.tokenize("she can't stop")
    assert "can" in result
    assert "not" in result
    # "t" isolé ne doit PAS apparaître (sinon l'ordre était mauvais)
    assert "t" not in result
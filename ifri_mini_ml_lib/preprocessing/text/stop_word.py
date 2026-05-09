class StopWordRemover:
    """
    Supprime les stop words d'une liste de tokens.

    Description:
        Les stop words sont des mots très fréquents dans une langue
        qui n'apportent pas de valeur informative pour l'analyse de texte
        (ex: "le", "la", "de", "et" en français).
        Cette classe permet de les identifier et les retirer d'une liste
        de tokens avant vectorisation.

    Args:
        language (str): Langue des stop words. Options: 'french', 'english'.
                        Default est 'english'.
        custom_stopwords (list, optional): Mots supplémentaires à considérer
                                           comme stop words. Default est None.

    Examples:
        >>> remover = StopWordRemover(language='english')
        >>> tokens = ["the", "film", "is", "really", "excellent"]
        >>> remover.transform(tokens)
        ['film', 'really', 'excellent']
    """

    BUILTIN_STOPWORDS = {
        "english": {
            "the", "a", "an", "is", "it", "in", "on", "at", "to",
            "for", "of", "and", "or", "but", "not", "with", "this",
            "that", "was", "are", "be", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may",
            "might", "shall", "can", "i", "you", "he", "she", "we",
            "they", "my", "your", "his", "her", "its", "our", "their",
            "from", "by", "as", "if", "so", "up", "out", "about",
            "into", "than", "then", "when", "there", "been", "me"
        },
        # TODO : ajouter les stop words français
        # "french": { ... }
    }

    def __init__(self, language="english", custom_stopwords=None):
        """
        Initialise le StopWordRemover.

        Args:
            language (str): Langue cible ('english' supporté pour l'instant).
            custom_stopwords (list, optional): À implémenter — liste de mots
                                               supplémentaires à traiter comme
                                               stop words.
        """
        self.language = language
        self.stopwords = set(self.BUILTIN_STOPWORDS.get(language, set()))
        # TODO : intégrer custom_stopwords dans self.stopwords

    def fit(self, X=None, y=None):
        """
        Aucun apprentissage nécessaire pour cette classe.
        Retourne self pour compatibilité avec les pipelines.

        Args:
            X: Ignoré.
            y: Ignoré.

        Returns:
            self
        """
        return self

    def transform(self, tokens):
        """
        Supprime les stop words d'une liste de tokens.

        Description:
            Accepte soit une liste simple de tokens (un seul document),
            soit une liste de listes (plusieurs documents).

        Args:
            tokens (list): Liste de tokens (str) ou liste de listes de tokens.

        Returns:
            list: Tokens filtrés, sans les stop words.
        """
        if not tokens:
            return []
        # Cas : liste de documents
        if isinstance(tokens[0], list):
            return [
                [t for t in doc if t.lower() not in self.stopwords]
                for doc in tokens
            ]
        # Cas : document unique
        return [t for t in tokens if t.lower() not in self.stopwords]

    def fit_transform(self, tokens, y=None):
        """
        Ajuste et transforme en une seule étape.

        Args:
            tokens (list): Liste de tokens ou liste de listes de tokens.
            y: Ignoré.

        Returns:
            list: Tokens filtrés sans les stop words.
        """
        return self.fit().transform(tokens)

    def get_stopwords(self):
        """
        Retourne l'ensemble des stop words utilisés.

        Returns:
            set: Ensemble des stop words actifs.
        """
        return self.stopwords
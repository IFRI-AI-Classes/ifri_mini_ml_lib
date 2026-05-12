class StopWordRemover:
    """
    Removes stop words from a list of tokens.

    Description:
        Stop words are very common words in a language
        that do not provide significangit addt informational value
        for text analysis (e.g., "the", "is", "and" in English).
        This class identifies and removes them from a list
        of tokens before vectorization.

    Args:
        language (str): Stop words language. Options: 'french', 'english'.
                        Default is 'english'.
        custom_stopwords (list, optional): Additional words to consider
                                           as stop words. Default is None.

    Examples:
        >>> remover = StopWordRemover(language='english')
        >>> tokens = ["the", "film", "is", "really", "excellent"]
        >>> remover.transform(tokens)
        ['film', 'really', 'excellent']
    """

    BUILTIN_STOPWORDS = {
        # Add English stop words
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

        # Add French stop words
        "french": {
            "le", "la", "les", "de", "du", "des", "un", "une",
            "et", "est", "en", "au", "aux", "ce", "qui", "que",
            "pour", "sur", "dans", "par", "avec", "il", "elle",
            "nous", "vous", "ils", "elles", "je", "tu", "on",
            "se", "sa", "son", "ses", "mon", "ma", "mes", "ton",
            "ta", "tes", "leur", "leurs", "y", "ne", "pas", "plus",
            "très", "bien", "comme", "aussi", "mais", "ou", "donc",
            "car", "si", "à", "été", "être", "avoir", "fait"
        },
    }

    def __init__(self, language="english", custom_stopwords=None):
        """
        Initializes the StopWordRemover.

        Args:
            language (str): Target language ('english' or 'french').
            custom_stopwords (list, optional): Additional words
                                               to treat as stop words.
        """
        self.language = language
        self.stopwords = set(self.BUILTIN_STOPWORDS.get(language, set()))

        if custom_stopwords:
            self.stopwords = self.stopwords.union(set(custom_stopwords))

    def fit(self, X=None, y=None):
        """
        No training is required for this class.
        Returns self for pipeline compatibility.

        Args:
            X: Ignored.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, tokens):
        """
        Removes stop words from a list of tokens.

        Description:
            Accepts either a single list of tokens (one document)
            or a list of token lists (multiple documents).

        Args:
            tokens (list): List of tokens (str) or list of token lists.

        Returns:
            list: Filtered tokens without stop words.
        """
        if not tokens:
            return []

        # Case: list of documents
        if isinstance(tokens[0], list):
            return [
                [t for t in doc if t.lower() not in self.stopwords]
                for doc in tokens
            ]

        # Case: single document
        return [t for t in tokens if t.lower() not in self.stopwords]

    def fit_transform(self, tokens, y=None):
        """
        Fits and transforms in a single step.

        Args:
            tokens (list): List of tokens or list of token lists.
            y: Ignored.

        Returns:
            list: Filtered tokens without stop words.
        """
        return self.fit().transform(tokens)

    def get_builtin_stopwords(self, language=None):
        """
        Returns the built-in stop words.

        Args:
            language (str, optional): Target language.
                                      If provided, returns stop words
                                      for that language only.

        Returns:
            dict or set:
                - Full dictionary of built-in stop words if no language is provided.
                - Set of stop words for the specified language otherwise.
        """
        if language:
            return self.BUILTIN_STOPWORDS.get(language, set())

        return self.BUILTIN_STOPWORDS

    def get_stopwords(self):
        """
        Returns the set of active stop words.

        Returns:
            set: Active stop words set.
        """
        return self.stopwords
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

        "english": {
            # Articles & determiners
            "a", "an", "the", "this", "that", "these", "those",
            "each", "every", "either", "neither", "any", "all",
            "both", "few", "more", "most", "other", "some", "such",
            "no", "own", "same", "than", "too", "very",
            # Personal pronouns
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "it", "its", "itself", "they", "them", "their", "theirs",
            "themselves", "what", "which", "who", "whom",
            # Auxiliary verbs
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing",
            "will", "would", "could", "should", "may", "might",
            "shall", "can", "need", "dare", "ought", "used",
            # Prepositions & conjunctions
            "at", "by", "for", "in", "of", "on", "to", "up",
            "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "out", "off", "over",
            "under", "again", "further", "then", "once",
            "and", "but", "or", "nor", "so", "yet",
            "if", "because", "while", "although", "though",
            "unless", "until", "since", "when", "where", "how",
            # Common adverbs
            "not", "with", "about", "against", "from", "there",
            "here", "why", "than", "just", "now", "also",
            "only", "same", "down", "however", "therefore",
            # Common verbs (low informational value)
            "get", "got", "go", "going", "gone", "went",
            "make", "made", "know", "think", "see", "come",
            "want", "look", "use", "find", "give", "tell",
            "work", "call", "try", "ask", "seem", "feel",
            "leave", "put", "mean", "keep", "let", "begin",
            "show", "hear", "play", "run", "move", "live",
            "believe", "hold", "bring", "happen", "write",
            "provide", "sit", "stand", "lose", "pay", "meet",
            "include", "continue", "set", "turn", "follows",
        },

        "french": {
            # Articles & déterminants
            "le", "la", "les", "un", "une", "des", "du", "de",
            "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta",
            "tes", "son", "sa", "ses", "notre", "votre", "leur",
            "nos", "vos", "leurs", "quel", "quelle", "quels", "quelles",
            "tout", "toute", "tous", "toutes", "autre", "autres",
            "même", "mêmes", "chaque", "plusieurs", "certains",
            "certaines", "quelques", "aucun", "aucune",
            # Pronoms personnels
            "je", "me", "moi", "tu", "te", "toi", "il", "lui",
            "elle", "nous", "vous", "ils", "elles", "on", "se",
            "soi", "y", "en", "qui", "que", "quoi", "dont", "où",
            "celui", "celle", "ceux", "celles", "ce", "ceci", "cela",
            # Verbes auxiliaires & copules
            "est", "sont", "était", "étaient", "être", "été",
            "avoir", "ai", "as", "avons", "avez", "ont", "avait",
            "avaient", "eu", "fait", "faire", "ferai", "fera",
            "sera", "seront", "serait", "seraient",
            # Prépositions & conjonctions
            "à", "au", "aux", "de", "du", "des", "en", "par",
            "pour", "sur", "sous", "dans", "avec", "sans", "entre",
            "vers", "chez", "contre", "dès", "depuis", "pendant",
            "avant", "après", "selon", "malgré", "sauf",
            "et", "ou", "mais", "donc", "or", "ni", "car",
            "si", "que", "quand", "comme", "lorsque", "puisque",
            "parce", "bien", "ainsi", "afin",
            # Adverbes courants
            "ne", "pas", "plus", "très", "aussi", "trop",
            "bien", "mal", "encore", "déjà", "toujours", "jamais",
            "souvent", "parfois", "ici", "là", "comment", "pourquoi",
            "oui", "non", "peut", "peu", "beaucoup", "moins",
            "même", "alors", "ainsi", "donc", "cependant",
            "toutefois", "pourtant", "néanmoins", "enfin",
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
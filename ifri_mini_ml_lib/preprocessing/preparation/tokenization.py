import re


class Tokenizer:
    """
    Text Tokenizer for NLP preprocessing tasks.

    Description:
        Implements a configurable text tokenizer capable of handling lowercase
        conversion, apostrophes, English contractions, punctuation removal and
        minimum token length filtering. Each preprocessing step can be enabled
        or disabled independently via constructor arguments.

    Args:
        lowercase (bool, optional): Convert all text to lowercase before tokenizing.
                                    Default is True.
        handle_apostrophes (bool, optional): Replace apostrophe characters with spaces.
                                             Default is True.
        handle_contractions (bool, optional): Expand English contractions
                                              (e.g. "can't" → "can not").
                                              Default is True.
        remove_punctuation (bool, optional): Strip punctuation characters from text.
                                             Default is True.
        min_token_length (int, optional): Minimum number of characters a token must
                                          have to be kept. Default is 1.

    Examples:
        >>> # Basic usage with default settings
        >>> tokenizer = Tokenizer()
        >>> tokenizer.tokenize("I can't do this!")
        ['i', 'can', 'not', 'do', 'this']

        >>> # Corpus tokenization
        >>> tokenizer = Tokenizer(lowercase=False, remove_punctuation=False)
        >>> tokenizer.tokenize_corpus(["Hello world", "It's great"])
        [['Hello', 'world'], ['It', 's', 'great']]
    """

    # ---------------------------------------------------------------------------
    # Mapping of English contractions to their expanded forms.
    # Keys are plain-text patterns (no regex); they are escaped at runtime.
    # The dictionary is iterated in insertion order, so more specific patterns
    # (e.g. "can't") are resolved before shorter suffix patterns (e.g. "'s").
    # ---------------------------------------------------------------------------
    CONTRACTION_MAP: dict[str, str] = {
        # Negative contractions
        "can't":    "can not",
        "won't":    "will not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "don't":    "do not",
        "didn't":   "did not",
        "isn't":    "is not",
        "aren't":   "are not",
        "wasn't":   "was not",
        "weren't":  "were not",
        # Auxiliary verb suffixes
        "'re":  " are",
        "'m":   " am",
        "'ll":  " will",
        "'d":   " would",
        "'ve":  " have",
        # Pronoun + 's → pronoun + is  (handled separately via regex group)
        # Possessive 's removal  (handled separately as a catch-all)
    }

    # Pronouns for which "'s" means "is" rather than a possessive marker.
    _PRONOUN_PATTERN: str = r"(it|he|she|that|what|who|here|there|how)"

    def __init__(
        self,
        lowercase: bool = True,
        handle_apostrophes: bool = True,
        handle_contractions: bool = True,
        remove_punctuation: bool = True,
        min_token_length: int = 1,
    ):
        """
        Initializes the Tokenizer with the desired preprocessing configuration.

        Args:
            lowercase (bool): Whether to convert text to lowercase.
            handle_apostrophes (bool): Whether to replace apostrophes with spaces.
            handle_contractions (bool): Whether to expand English contractions.
            remove_punctuation (bool): Whether to remove punctuation characters.
            min_token_length (int): Minimum token length to retain after splitting.
        """
        self.lowercase = lowercase
        self.handle_apostrophes = handle_apostrophes
        self.handle_contractions = handle_contractions
        self.remove_punctuation = remove_punctuation
        self.min_token_length = min_token_length


    # ------------------------------------------------------------------

    def tokenize_corpus(self, corpus: list[str]) -> list[list[str]]:
        """
        Tokenizes every document in a corpus.

        Description:
            Applies the full tokenization pipeline to each document in the
            provided corpus and returns a list of token lists, one per document.

        Args:
            corpus (list[str]): A list of raw text documents to tokenize.

        Returns:
            list[list[str]]: A list where each element is the token list
                             produced by tokenize() for the corresponding document.
        """
        return [self.tokenize(doc) for doc in corpus]

    def tokenize(self, text: str) -> list[str]:
        """
        Runs the full preprocessing pipeline on a single text string.

        Description:
            Applies each enabled preprocessing step in the following order:
            lowercase → contractions → apostrophes → punctuation removal → split
            → length filtering. Steps are skipped when their corresponding flag
            is set to False.

        Args:
            text (str): The raw input string to tokenize.

        Returns:
            list[str]: Ordered list of tokens extracted from the input text.
        """
        if self.lowercase:
            text = self.lower_case(text)
        if self.handle_contractions:
            text = self._handle_contractions(text)
        if self.handle_apostrophes:
            text = self._handle_apostrophes(text)
        if self.remove_punctuation:
            text = self._remove_punctuation(text)

        tokens = text.split()

        if self.min_token_length > 0:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return tokens

    # ------------------------------------------------------------------

    def lower_case(self, text: str) -> str:
        """
        Converts all characters in the text to lowercase.

        Args:
            text (str): Input string.

        Returns:
            str: Lowercased version of the input string.
        """
        return text.lower()

    def _handle_contractions(self, text: str) -> str:
        """
        Expands English contractions using a dictionary-based lookup.

        Description:
            Iterates over CONTRACTION_MAP and replaces each contraction with its
            expanded form using a single re.sub() call per entry.  After the
            dictionary pass, two special cases are resolved with dedicated patterns:
              1. Pronoun + "'s"  →  pronoun + " is"  (e.g. "it's" → "it is").
              2. Remaining "'s"  →  empty string      (possessive stripping).

            Using a dictionary instead of hard-coded successive re.sub() calls
            makes it trivial to add, remove, or modify contractions without
            touching the method body.

        Args:
            text (str): Input string possibly containing contractions.

        Returns:
            str: Text with all recognised contractions expanded.
        """
        for contraction, expansion in self.CONTRACTION_MAP.items():
            # re.escape ensures that punctuation in the key (e.g. apostrophe)
            # is treated as a literal character, not a regex metacharacter.
            text = re.sub(re.escape(contraction), expansion, text)

        # Special case 1 — pronoun + 's → pronoun + is
        text = re.sub(self._PRONOUN_PATTERN + r"'s", r"\1 is", text)

        # Special case 2 — catch-all possessive 's removal
        text = re.sub(r"'s\b", "", text)

        return text

    def _handle_apostrophes(self, text: str) -> str:
        """
        Replaces apostrophe characters with spaces.

        Description:
            Handles both the ASCII apostrophe (') and the Unicode right single
            quotation mark (U+2019, '\u2019') commonly introduced by smart-quote
            processing. This step should run *after* contraction expansion so
            that apostrophes used in contractions are already consumed.

        Args:
            text (str): Input string possibly containing apostrophe characters.

        Returns:
            str: Text with apostrophes replaced by single spaces.
        """
        return re.sub(r"['\u2019]", " ", text)

    def _remove_punctuation(self, text: str) -> str:
        """
        Strips punctuation characters from the text.

        Description:
            Replaces any character that is not an ASCII letter, an accented
            Latin character (À–ÿ), a digit, or a whitespace character with a
            single space. This preserves multilingual tokens while removing all
            punctuation marks and special symbols.

        Args:
            text (str): Input string possibly containing punctuation.

        Returns:
            str: Text with punctuation replaced by spaces.
        """
        return re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    


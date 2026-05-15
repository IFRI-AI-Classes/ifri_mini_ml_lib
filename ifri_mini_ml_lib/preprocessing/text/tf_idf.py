import math
import numpy as np  


class TFIDFVectorizer:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer.

    Description:
        Transforms tokenized text documents into numerical TF-IDF vectors.
        Each dimension of the output vector corresponds to a term in the learned
        vocabulary, and its value reflects how important that term is in a given
        document relative to the entire corpus.

        The implementation matches scikit-learn's TfidfVectorizer behaviour when
        smooth_idf=True and norm='l2' (default settings), producing an identical
        output matrix on the same tokenized input.

        The pipeline is composed of three stages:
          1. fit()          — builds the vocabulary and computes IDF weights.
          2. transform()    — converts tokenized documents into TF-IDF vectors.
          3. fit_transform() — convenience wrapper that runs both in one call.

        Optional n-gram support allows the vectorizer to capture multi-word
        expressions (e.g. "not good") that single-token models would miss.

    Args:
        smooth_idf (bool, optional): Add 1 to numerator and denominator of the
                                     IDF formula to avoid zero-division and to
                                     prevent IDF from being zero for terms that
                                     appear in every document.
                                     Formula: log((N+1)/(df+1)) + 1.
                                     Default is True.
        norm (str or None, optional): Vector normalization applied after scoring.
                                      Supported value: "l2" (unit-norm rows).
                                      Pass None to skip normalization.
                                      Default is "l2".
        sublinear_tf (bool, optional): Apply sublinear TF scaling:
                                       tf = 1 + log(tf) instead of raw frequency.
                                       Reduces the influence of very frequent terms.
                                       Default is False.
        ngram_range (tuple, optional): The lower and upper boundary of the range
                                       of n-values for n-grams to be extracted.
                                       (1, 1) means unigrams only.
                                       (1, 2) means unigrams and bigrams.
                                       Default is (1, 1).
        binary (bool, optional): If True, all non-zero term counts are set to 1.
                                  Encodes term presence rather than frequency.
                                  Default is False.

    Attributes:
        vocabulary_ (dict): Mapping of term → column index, built during fit().
        idf_ (dict): Mapping of term → IDF weight, built during fit().

    Examples:
        >>> # Basic usage with default settings
        >>> tfidf = TFIDFVectorizer()
        >>> corpus = [["cat", "eats", "fish"], ["dog", "runs", "fast"]]
        >>> tfidf.fit(corpus)
        >>> matrix = tfidf.transform(corpus)

        >>> # One-shot fit_transform with bigrams
        >>> tfidf = TFIDFVectorizer(ngram_range=(1, 2))
        >>> matrix = tfidf.fit_transform(corpus)

        >>> # Binary mode — presence/absence only
        >>> tfidf = TFIDFVectorizer(binary=True, norm=None)
        >>> matrix = tfidf.fit_transform(corpus)

        >>> # Sublinear TF scaling, no smoothing, no normalization
        >>> tfidf = TFIDFVectorizer(smooth_idf=False, norm=None, sublinear_tf=True)
        >>> matrix = tfidf.fit_transform(corpus)
    """

    def __init__(
        self,
        smooth_idf: bool = True,
        norm: str = "l2",
        sublinear_tf: bool = False,
        ngram_range: tuple = (1, 1),
        binary: bool = False,
        min_df: int = 1,
        max_features: int = None,
    ):
        """
        Initializes the TFIDFVectorizer with the desired configuration.

        Args:
            smooth_idf (bool): Whether to apply IDF smoothing.
            norm (str or None): Normalization scheme to apply to output vectors.
            sublinear_tf (bool): Whether to apply sublinear TF scaling.
            ngram_range (tuple): Range (min_n, max_n) for n-gram extraction.
            binary (bool): Whether to binarize term frequencies.
        """
        self.smooth_idf = smooth_idf
        self.norm = norm
        self.sublinear_tf = sublinear_tf
        self.ngram_range = ngram_range
        self.binary = binary
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_ = {}
        self.idf_ = {}

    # ------------------------------------------------------------------
    #  N-grams
    # ------------------------------------------------------------------

    def _generate_ngrams(self, tokens: list) -> list:
        """
        Generates n-grams from a list of tokens.

        Description:
            Applies a sliding window of size n over the token list to produce
            all contiguous sequences of n tokens. The window size ranges from
            ngram_range[0] to ngram_range[1] inclusive.

            Example with ngram_range=(1, 2):
              tokens = ["cat", "eats", "fish"]
              → ["cat", "eats", "fish", "cat eats", "eats fish"]

        Args:
            tokens (list[str]): List of tokens from a single document.

        Returns:
            list[str]: List of all n-grams generated from the input tokens.
        """
        all_ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i + n])
                all_ngrams.append(ngram)
        return all_ngrams

    # ------------------------------------------------------------------
    #  TF
    # ------------------------------------------------------------------

    def _compute_tf(self, doc: list) -> dict:
        """
        Computes the term frequency for each term in a document.

        Description:
            First counts raw occurrences of each term, then divides by the
            total number of terms to obtain relative frequencies.
            If sublinear_tf is enabled, applies the transformation
            tf = 1 + log(raw_count) to dampen the effect of very frequent terms.
            If binary is enabled, all non-zero frequencies are set to 1.

        Args:
            doc (list[str]): List of terms (tokens or n-grams) for one document.

        Returns:
            dict: Mapping of term → TF score for the given document.
        """
        tf = {}
        total = len(doc)
        for term in doc:
            tf[term] = tf.get(term, 0) + 1

        for term in tf:
            if self.binary:
                tf[term] = 1
            elif self.sublinear_tf:
                tf[term] = 1 + math.log(tf[term])
            else:
                tf[term] = tf[term] / total

        return tf

    # ------------------------------------------------------------------
    #  IDF
    # ------------------------------------------------------------------

    def _compute_idf(self, corpus: list) -> dict:
        """
        Computes the Inverse Document Frequency for each term in the corpus.

        Description:
            Counts in how many documents each term appears (document frequency),
            then applies the IDF formula.

            Two formulas are supported:
              - Smooth (default): log((N+1) / (df+1)) + 1
                  Avoids zero-division and ensures IDF >= 1 for all terms.
              - Raw:              log(N / df)
                  Classic formulation; IDF = 0 for terms in every document.

            Note: set(doc) is used when counting document frequencies to ensure
            each term is counted at most once per document, regardless of how
            many times it appears within that document.

        Args:
            corpus (list[list[str]]): List of tokenized documents.

        Returns:
            dict: Mapping of term → IDF weight.
        """
        N = len(corpus)
        df = {}
        for doc in corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        idf = {}
        for term, freq in df.items():
            if self.smooth_idf:
                idf[term] = math.log((N + 1) / (freq + 1)) + 1
            else:
                idf[term] = math.log(N / freq)
        return idf

    # ------------------------------------------------------------------
    #  Normalisation L2
    # ------------------------------------------------------------------

    def _normalize_l2(self, matrix: list) -> list:
        """
        Applies L2 normalization to each row of the TF-IDF matrix.

        Description:
            Divides each element of a row by the row's Euclidean norm (L2 norm),
            so that the resulting vector has unit length. This makes document
            vectors comparable regardless of document length.

            Formula: normalized[i] = row[i] / sqrt(sum(row[j]^2 for all j))

            Rows with a zero norm (empty documents) are returned unchanged
            to avoid division by zero.

        Args:
            matrix (list[list[float]]): The raw TF-IDF matrix to normalize.

        Returns:
            list[list[float]]: Matrix with each row normalized to unit length.
        """
        normalized = []
        for row in matrix:
            norm = math.sqrt(sum(x ** 2 for x in row))
            if norm == 0:
                normalized.append(row[:])
            else:
                normalized.append([x / norm for x in row])
        return normalized

    # ------------------------------------------------------------------
    #  Fit
    # ------------------------------------------------------------------

    def fit(self, corpus: list) -> None:
        """
        Learns the vocabulary and IDF weights from a tokenized corpus.

        Description:
            Collects all unique terms across the corpus (after n-gram generation),
            sorts them alphabetically to produce a deterministic column ordering
            that matches scikit-learn's behaviour, and assigns each term a unique
            column index. Then computes and stores the IDF weight for every term.

            After fit() is called, vocabulary_ and idf_ are populated and
            transform() can be called on any tokenized corpus.

        Args:
            corpus (list[list[str]]): List of tokenized documents. Each document
                                      is a list of string tokens produced by
                                      Tokenizer.tokenize() or equivalent.

        Returns:
            None
        """
        expanded_corpus = [self._generate_ngrams(doc) for doc in corpus]

        idf_temp, df = self._compute_idf(expanded_corpus)

        terms_valides = {term for term, freq in df.items() if freq >= self.min_df}

        if self.max_features is not None:
            terms_valides = set(
                sorted(terms_valides, key=lambda t: idf_temp.get(t, 0), reverse=True)
                [:self.max_features]
            )

        for index, term in enumerate(sorted(terms_valides)):
            self.vocabulary_[term] = index

        self.idf_ = {term: idf_temp[term] for term in self.vocabulary_}

    # ------------------------------------------------------------------
    #  Transform
    # ------------------------------------------------------------------

    def transform(self, corpus: list) -> list:
        """
        Converts tokenized documents into TF-IDF vectors using the learned vocabulary.

        Description:
            For each document, generates n-grams, computes TF scores, multiplies
            by the corresponding IDF weights, and places each score at the column
            index defined by vocabulary_. Terms not seen during fit() are silently
            ignored (out-of-vocabulary handling).

            If norm="l2", the entire matrix is L2-normalized before being returned.

        Args:
            corpus (list[list[str]]): List of tokenized documents to transform.
                                      Must use the same tokenization scheme as
                                      the corpus passed to fit().

        Returns:
            list[list[float]]: TF-IDF matrix of shape
                               (n_documents, n_vocabulary_terms).
                               Each row is the TF-IDF vector for one document.
        """
        matrix = []
        for doc in corpus:
            ngrams = self._generate_ngrams(doc)
            tf = self._compute_tf(ngrams)
            scores = [0.0] * len(self.vocabulary_)
            for term, freq in tf.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    scores[idx] = freq * self.idf_.get(term, 0)
            matrix.append(scores)

        if self.norm == "l2":
            matrix = self._normalize_l2(matrix)
        return matrix

    # ------------------------------------------------------------------
    #  Fit-Transform
    # ------------------------------------------------------------------

    def fit_transform(self, corpus: list) -> list:
        """
        Fits the vectorizer on the corpus and transforms it in one step.

        Description:
            Convenience method that calls fit() followed by transform() on the
            same corpus. Equivalent to calling both methods separately but avoids
            iterating over the corpus twice in user code.

        Args:
            corpus (list[list[str]]): List of tokenized documents.

        Returns:
            list[list[float]]: TF-IDF matrix — same output as transform(corpus)
                               called after fit(corpus).
        """
        self.fit(corpus)
        return self.transform(corpus)


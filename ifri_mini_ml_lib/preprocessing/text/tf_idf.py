import math
import numpy as np

class TF_IDF:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) vectorizer.

    Description:
        Implements the TF-IDF weighting scheme used in information retrieval
        and NLP to represent text documents as numerical vectors. Each dimension
        of the output vector corresponds to a word in the learned vocabulary,
        and its value reflects how important that word is in a given document
        relative to the entire corpus.

        The pipeline is composed of three stages:
          1. fit()       — builds the vocabulary and computes IDF weights.
          2. transform() — converts tokenized documents into TF-IDF vectors.
          3. fit_transform() — convenience wrapper that runs both in one call.

    Args:
        smooth_idf (bool, optional): Add 1 to numerator and denominator of the
                                     IDF formula to avoid zero-division when a
                                     term appears in every document.
                                     Default is True.
        norm (str, optional): Vector normalization to apply after scoring.
                              Supported value: "l2" (unit-norm rows).
                              Pass None to skip normalization.
                              Default is "l2".
        sublinear_tf (bool, optional): Apply sublinear TF scaling:
                                       tf = 1 + log(tf) instead of raw frequency.
                                       Reduces the weight of very frequent terms.
                                       Default is False.

    Attributes:
        vocabulary_ (dict): Mapping of term → column index, built during fit().
        idf_ (dict): Mapping of term → IDF weight, built during fit().

    Examples:
        >>> # Basic fit + transform
        >>> tfidf = TF_IDF()
        >>> corpus = [["hello", "world"], ["hello", "numpy"]]
        >>> tfidf.fit(corpus)
        >>> matrix = tfidf.transform(corpus)

        >>> # One-shot fit_transform
        >>> tfidf = TF_IDF(smooth_idf=False, norm=None, sublinear_tf=True)
        >>> matrix = tfidf.fit_transform(corpus)
    """
    def __init__(self, smooth_idf=True, norm="l2", sublinear_tf=False):
        self.smooth_idf = smooth_idf
        self.norm = norm
        self.sublinear_tf = sublinear_tf
        self.vocabulary_ = {}
        self.idf_ = {}

    def _compute_tf(self, doc):
        tf = {}
        total = len(doc)
        for mot in doc:
            tf[mot] = tf.get(mot, 0) + 1
        for mot in tf:
            raw = tf[mot] / total
            tf[mot] = (1 + math.log(tf[mot])) if self.sublinear_tf else raw
        return tf

    def _compute_idf(self, corpus):
        N = len(corpus)
        df = {}  
        for doc in corpus:
            for mot in set(doc):
                df[mot] = df.get(mot, 0) + 1

        idf = {}
        for mot, freq in df.items():
            if self.smooth_idf:
                idf[mot] = math.log((N + 1) / (freq + 1)) + 1
            else:
                idf[mot] = math.log(N / freq)
        return idf

 
    def _normalize_l2(self, matrix):
        normalized = []
        for row in matrix:
            norme = math.sqrt(sum(x ** 2 for x in row))
            if norme == 0:
                normalized.append(row[:])
            else:
                normalized.append([x / norme for x in row])
        return normalized


    def fit(self, corpus):
        mots_uniques = set()
        for doc in corpus:
            mots_uniques.update(doc)

        for index, mot in enumerate(sorted(mots_uniques)):
            self.vocabulary_[mot] = index

        self.idf_ = self._compute_idf(corpus)


    def transform(self, corpus):
        matrix = []
        for doc in corpus:
            tf = self._compute_tf(doc)
            scores = [0.0] * len(self.vocabulary_)
            for mot, freq in tf.items():
                if mot in self.vocabulary_:
                    idx = self.vocabulary_[mot]
                    scores[idx] = freq * self.idf_.get(mot, 0)
            matrix.append(scores)

        if self.norm == "l2":
            matrix = self._normalize_l2(matrix)
        return matrix

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)


import math


class CountVectorizer:
    def __init__(self, binary=False, ngram_range=(1, 1), stop_words=None):
        """
        Composant de base pour transformer du texte en fréquences numériques.
        """
        self.binary = binary
        self.ngram_range = ngram_range
        self.stop_words = stop_words if stop_words else []
        self.vocabulary_ = {}
        self.feature_names_ = []

    def _generate_ngrams(self, tokens):
        """ Génère des n-grams à partir d'une liste de tokens. """
        all_ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i + n])
                all_ngrams.append(ngram)
        return all_ngrams

    def fit(self, raw_documents):
        """
        Apprend le vocabulaire à partir des documents tokenisés.
        """
        unique_terms = set()
        for doc in raw_documents:
            # Filtrage des stop_words
            filtered = [t for t in doc if t not in self.stop_words]
            terms = self._generate_ngrams(filtered)
            unique_terms.update(terms)

        self.feature_names_ = sorted(list(unique_terms))
        self.vocabulary_ = {term: i for i, term in enumerate(self.feature_names_)}
        return self

    def transform(self, raw_documents, return_sparse=False):
        """
        Transforme les documents en vecteurs.
        :param return_sparse: Si True, retourne une liste de dictionnaires.
        """
        matrix = []
        for doc in raw_documents:
            # On utilise un dictionnaire temporaire pour compter les occurrences dans le doc
            doc_counts = {}
            filtered = [t for t in doc if t not in self.stop_words]
            terms = self._generate_ngrams(filtered)

            for term in terms:
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    if self.binary:
                        doc_counts[idx] = 1
                    else:
                        doc_counts[idx] = doc_counts.get(idx, 0) + 1

            if return_sparse:
                matrix.append(doc_counts)
            else:
                # Conversion en format dense (liste de zéros)
                vector = [0] * len(self.vocabulary_)
                for idx, count in doc_counts.items():
                    vector[idx] = count
                matrix.append(vector)

        return matrix

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)
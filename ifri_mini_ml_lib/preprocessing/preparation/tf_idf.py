import math

class TF_IDF:
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


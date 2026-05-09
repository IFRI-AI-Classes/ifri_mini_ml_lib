import math

class TF_IDF:
    def __init__(self):
        self.vocabulary_ = {}  
        self.idf_ = {}         

    def _compute_tf(self, doc):
        tf = {}
        total = len(doc)
        for mot in doc:
            if mot in tf:
                tf[mot] += 1
            else:
                tf[mot] = 1
        for mot in tf:
            tf[mot] = tf[mot] / total
        return tf

    def _compute_idf(self, corpus):
        N = len(corpus)
        idf = {}
        for doc in corpus:
            for mot in set(doc):  
                if mot in idf:
                    idf[mot] += 1
                else:
                    idf[mot] = 1
        for mot in idf:
            idf[mot] = math.log(N / idf[mot])
        return idf

    def fit(self, corpus):
        index = 0
        for doc in corpus:
            for mot in doc:
                if mot not in self.vocabulary_:
                    self.vocabulary_[mot] = index
                    index += 1
        self.idf_ = self._compute_idf(corpus)

    def transform(self, corpus):
        matrix = []
        for doc in corpus:
            tf = self._compute_tf(doc)
            scores = [0.0] * len(self.vocabulary_)
            for mot, freq in tf.items():
                if mot in self.vocabulary_:
                    index = self.vocabulary_[mot]
                    idf = self.idf_.get(mot, 0)
                    scores[index] = freq * idf
            matrix.append(scores)
        return matrix

    def fit_transform(self, corpus):

        self.fit(corpus)
        return self.transform(corpus)


import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = np.array(x)
        self.y_train = np.array(y)

    def predict(self, X):
        return [self._predict(x) for x in X]
    
    def _predict(self, x):
        distances = []
        for train_point in self.x_train:
            distances.append(np.sqrt(np.sum((x - train_point)**2)))
            
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices] 

        label_counts = {}
        for label in k_nearest_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            
        return label_counts
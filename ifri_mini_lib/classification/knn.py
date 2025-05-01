import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.x_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        # Calculer les distances avec np.linalg.norm
        distances = np.linalg.norm(self.x_train - x, axis=1)

        # Trouver les indices des k plus proches voisins
        k_indices = np.argsort(distances)[:self.k]

        # Récupérer les étiquettes des k plus proches voisins
        k_nearest_labels = self.y_train[k_indices]

        # Créer un compteur "from scratch" pour les étiquettes
        label_counts = {}
        for label in k_nearest_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Retourner l'étiquette majoritaire
        return max(label_counts, key=label_counts.get)

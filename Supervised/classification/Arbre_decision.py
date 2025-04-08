import numpy as np
from collections import Counter

class DecisionTree:
    """
    Implémentation simple d'un arbre de décision pour la classification binaire ou multi-classe.

    :param max_depth: Profondeur maximale de l'arbre (None pour illimité)
    :type max_depth: int or None
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        """
        Entraîne l'arbre de décision sur les données.

        :param X: Données d'entrée de forme (n_samples, n_features)
        :type X: np.ndarray
        :param y: Étiquettes associées de forme (n_samples,)
        :type y: np.ndarray
        :param depth: Profondeur actuelle (utilisée en récursivité)
        :type depth: int
        :return: Structure récursive de l'arbre (dictionnaire ou classe majoritaire)
        :rtype: dict or int
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if (self.max_depth is not None and depth == self.max_depth) or (n_classes == 1):
            return self._most_common_label(y)

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return self._most_common_label(y)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_subtree = self.fit(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.fit(X[right_indices], y[right_indices], depth + 1)

        if depth == 0:
            self.tree = {
                "feature_index": best_feature,
                "threshold": best_threshold,
                "left": left_subtree,
                "right": right_subtree
            }

        return {
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def _best_split(self, X, y):
        """
        Trouve la meilleure séparation (feature, seuil) qui maximise le gain d'information.

        :param X: Données d'entrée
        :type X: np.ndarray
        :param y: Étiquettes
        :type y: np.ndarray
        :return: Index de la feature et seuil optimal
        :rtype: tuple(int, float)
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_index, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature_index, threshold):
        """
        Calcule le gain d'information pour une séparation donnée.

        :param X: Données d'entrée
        :type X: np.ndarray
        :param y: Étiquettes
        :type y: np.ndarray
        :param feature_index: Index de la feature testée
        :type feature_index: int
        :param threshold: Seuil utilisé pour la séparation
        :type threshold: float
        :return: Gain d'information
        :rtype: float
        """
        parent_entropy = self._entropy(y)

        left_indices = X[:, feature_index] < threshold
        right_indices = X[:, feature_index] >= threshold

        if sum(left_indices) == 0 or sum(right_indices) == 0:
            return 0

        n = len(y)
        n_left, n_right = sum(left_indices), sum(right_indices)
        e_left = self._entropy(y[left_indices])
        e_right = self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        return parent_entropy - child_entropy

    def _entropy(self, y):
        """
        Calcule l'entropie d'un ensemble d'étiquettes.

        :param y: Étiquettes
        :type y: np.ndarray
        :return: Entropie de Shannon
        :rtype: float
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _most_common_label(self, y):
        """
        Retourne la classe la plus fréquente dans un ensemble.

        :param y: Étiquettes
        :type y: np.ndarray
        :return: Classe majoritaire
        :rtype: int
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Prédit les classes pour un ensemble d'exemples.

        :param X: Données d'entrée (n_samples, n_features)
        :type X: np.ndarray
        :return: Prédictions pour chaque exemple
        :rtype: np.ndarray
        """
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        """
        Prédit la classe pour un seul exemple.

        :param x: Exemple unique (1D)
        :type x: np.ndarray
        :param tree: Arbre de décision (sous-forme de dictionnaire récursif)
        :type tree: dict or int
        :return: Classe prédite
        :rtype: int
        """
        if not isinstance(tree, dict):
            return tree

        feature_index = tree["feature_index"]
        threshold = tree["threshold"]

        if x[feature_index] < threshold:
            return self._predict_single(x, tree["left"])
        else:
            return self._predict_single(x, tree["right"])

    def print_tree(self, tree=None, indent=" "):
        """
        Affiche l'arbre de manière textuelle simple.

        :param tree: Arbre à afficher (par défaut, l'arbre principal)
        :type tree: dict or int
        :param indent: Indentation visuelle
        :type indent: str
        """
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            print(indent + "Classe:", tree)
            return

        print(indent + "Feature", tree["feature_index"], "<", tree["threshold"])
        print(indent + "--> True:")
        self.print_tree(tree["left"], indent + "  ")
        print(indent + "--> False:")
        self.print_tree(tree["right"], indent + "  ")

    def print_visual_tree(self, tree=None, indent="", last='updown'):
        """
        Affiche l'arbre de décision de manière visuelle avec indentation.

        :param tree: Arbre à afficher
        :type tree: dict or int
        :param indent: Indentation visuelle
        :type indent: str
        :param last: Position dans l'arbre ('left', 'right', 'updown')
        :type last: str
        """
        if tree is None:
            tree = self.tree
        
        if not isinstance(tree, dict):
            print(indent + "+-- " + f"Classe: {tree}")
            return

        print(indent + "+-- " + f"Feature {tree['feature_index']} < {tree['threshold']}?")
        
        self.print_visual_tree(tree["left"], indent + "│   ", 'left')
        self.print_visual_tree(tree["right"], indent + "    ", 'right')

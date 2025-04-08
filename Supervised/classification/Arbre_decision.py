import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
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

        return {"feature_index": best_feature, "threshold": best_threshold,
                "left": left_subtree, "right": right_subtree}

    def _best_split(self, X, y):
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
        parent_entropy = self._entropy(y)

        left_indices = X[:, feature_index] < threshold
        right_indices = X[:, feature_index] >= threshold

        if sum(left_indices) == 0 or sum(right_indices) == 0:
            return 0

        n = len(y)
        n_left, n_right = sum(left_indices), sum(right_indices)
        e_left, e_right = self._entropy(y[left_indices]), self._entropy(y[right_indices])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        feature_index = tree["feature_index"]
        threshold = tree["threshold"]

        if x[feature_index] < threshold:
            return self._predict_single(x, tree["left"])
        else:
            return self._predict_single(x, tree["right"])

    def print_tree(self, tree=None, indent=" "):
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
        if tree is None:
            tree = self.tree
        
        if not isinstance(tree, dict):
            print(indent + "+-- " + f"Classe: {tree}")
            return

        print(indent + "+-- " + f"Feature {tree['feature_index']} < {tree['threshold']}?")
        
        indent_add = "    " if last == 'updown' else "│   "
        
        # Branche True
        self.print_visual_tree(tree["left"], indent + "│   ", 'left')
        # Branche False 
        self.print_visual_tree(tree["right"], indent + "    ", 'right')
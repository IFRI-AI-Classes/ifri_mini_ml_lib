import numpy as np
from collections import Counter

class DecisionTree:
    """
    Simple implementation of a decision tree for binary or multi-class classification.

    :param max_depth: Maximum depth of the tree (None for unlimited depth)
    :type max_depth: int or None
    """

    def __init__(self, max_depth=None):
        """
        Initialize the decision tree with a specified maximum depth.

        :param max_depth: Maximum depth of the tree (None for unlimited depth)
        :type max_depth: int or None
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        """
        Train the decision tree on the provided data.

        :param X: Input features of shape (n_samples, n_features)
        :type X: np.ndarray
        :param y: Corresponding labels of shape (n_samples,)
        :type y: np.ndarray
        :param depth: Current depth (used for recursion)
        :type depth: int
        :return: Recursive tree structure (dictionary or majority class)
        :rtype: dict or int
        
        Example:
            tree = DecisionTree(max_depth=3)
            tree.fit(X_train, y_train)
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stop if max depth is reached or all samples belong to the same class
        if (self.max_depth is not None and depth == self.max_depth) or (n_classes == 1):
            return self._most_common_label(y)

        best_feature, best_threshold = self._best_split(X, y)

        # If no valid split is found
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
        Find the best feature and threshold to split on that maximizes information gain.

        :param X: Input features
        :type X: np.ndarray
        :param y: Labels
        :type y: np.ndarray
        :return: Index of the best feature and optimal threshold
        :rtype: tuple(int, float)
        
        Example:
            feature_index, threshold = tree._best_split(X_train, y_train)
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
        Calculate the information gain for a given split.

        :param X: Input features
        :type X: np.ndarray
        :param y: Labels
        :type y: np.ndarray
        :param feature_index: Index of the feature to split on
        :type feature_index: int
        :param threshold: Threshold value to split the feature
        :type threshold: float
        :return: Information gain value
        :rtype: float
        
        Example:
            gain = tree._information_gain(X_train, y_train, feature_index, threshold)
        """
        parent_entropy = self._entropy(y)

        left_indices = X[:, feature_index] < threshold
        right_indices = X[:, feature_index] >= threshold

        # Avoid division by zero or invalid splits
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
        Compute Shannon entropy for a label set.

        :param y: Labels
        :type y: np.ndarray
        :return: Entropy value
        :rtype: float
        
        Example:
            entropy = tree._entropy(y_train)
        """
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _most_common_label(self, y):
        """
        Return the most frequent label in a set.

        :param y: Labels
        :type y: np.ndarray
        :return: Most common class label
        :rtype: int
        
        Example:
            most_common = tree._most_common_label(y_train)
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Predict class labels for given input samples.

        :param X: Input features (n_samples, n_features)
        :type X: np.ndarray
        :return: Predicted class labels
        :rtype: np.ndarray
        
        Example:
            predictions = tree.predict(X_test)
        """
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        """
        Predict the class label for a single sample.

        :param x: Single input example (1D array)
        :type x: np.ndarray
        :param tree: Decision tree (recursively structured dict)
        :type tree: dict or int
        :return: Predicted class
        :rtype: int
        
        Example:
            prediction = tree._predict_single(X_test[0], tree)
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
        Print a simple textual representation of the tree.

        :param tree: Tree to print (default is the main tree)
        :type tree: dict or int
        :param indent: Visual indentation string
        :type indent: str
        
        Example:
            tree.print_tree()
        """
        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            print(indent + "Class:", tree)
            return

        print(indent + f"Feature {tree['feature_index']} < {tree['threshold']}")
        print(indent + "--> True:")
        self.print_tree(tree["left"], indent + "  ")
        print(indent + "--> False:")
        self.print_tree(tree["right"], indent + "  ")

    def print_visual_tree(self, tree=None, indent="", last='updown'):
        """
        Visually print the tree structure with indentation and branches.

        :param tree: Tree to print
        :type tree: dict or int
        :param indent: Indentation for formatting
        :type indent: str
        :param last: Tree position indicator ('left', 'right', 'updown')
        :type last: str
        
        Example:
            tree.print_visual_tree()
        """
        if tree is None:
            tree = self.tree
        
        if not isinstance(tree, dict):
            print(indent + "+-- " + f"Class: {tree}")
            return

        print(indent + "+-- " + f"Feature {tree['feature_index']} < {tree['threshold']}?")
        
        self.print_visual_tree(tree["left"], indent + "│   ", 'left')
        self.print_visual_tree(tree["right"], indent + "    ", 'right')

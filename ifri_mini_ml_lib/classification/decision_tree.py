import numpy as np
from collections import Counter

class DecisionTree:
    """
    A decision tree classifier with regularization parameters to prevent overfitting.
    
    Description:
    This implementation supports both classification tasks using information gain
    (entropy) as the splitting criterion. It includes several regularization
    techniques like max depth, minimum samples per split, and minimum samples per leaf.
    
    Args:
        max_depth (int, optional): Maximum depth of the tree. Defaults to None (no limit).
        min_samples_split (int): Minimum number of samples required to split a node. Defaults to 2.
        min_samples_leaf (int): Minimum number of samples required at each leaf node. Defaults to 1.
        min_impurity_decrease (float): Minimum impurity decrease required for a split. Defaults to 0.0.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.tree = None

    def fit(self, X, y, depth=0):
        """
        Builds the decision tree from the training data (X, y).
        
        Description:
        Recursively builds the decision tree by finding the best splits according to 
        information gain while respecting the regularization constraints.
        
        Args:
            X (ndarray): Training data of shape (n_samples, n_features)
            y (ndarray): Target values of shape (n_samples,)
            depth (int): Current depth of the tree (used internally during recursion)
            
        Returns:
            dict: The constructed decision tree node (either a decision node or a leaf value)
        """
        n_samples = X.shape[0] if len(X.shape) > 1 else 1
        
        # Check stopping conditions
        if (n_samples < self.min_samples_split or 
            (self.max_depth is not None and depth >= self.max_depth) or
            len(np.unique(y)) == 1):
            self.tree = self._most_common_label(y)
            return self.tree

        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_feature is None or best_gain < self.min_impurity_decrease:
            self.tree = self._most_common_label(y)
            return self.tree

        # Split data
        left_mask = X[:, best_feature] < best_threshold
        right_mask = ~left_mask
        
        # Check leaf constraints
        if (np.sum(left_mask) < self.min_samples_leaf or 
            np.sum(right_mask) < self.min_samples_leaf):
            self.tree = self._most_common_label(y)
            return self.tree

        # Recursively build the tree
        self.tree = {
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left": self.fit(X[left_mask], y[left_mask], depth + 1),
            "right": self.fit(X[right_mask], y[right_mask], depth + 1)
        }
        
        return self.tree

    def _best_split(self, X, y):
        """
        Finds the best feature and threshold to split on.
        
        Description:
        Evaluates all possible splits for each feature and returns the one that
        provides the highest information gain.
        
        Args:
            X (ndarray): Input features of shape (n_samples, n_features)
            y (ndarray): Target values of shape (n_samples,)
            
        Returns:
            tuple: (best_feature_index, best_threshold, best_gain)
                   Returns (None, None, -inf) if no valid split is found
        """
        best_gain = -np.inf
        best_feature, best_threshold = None, None
        parent_entropy = self._entropy(y)

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            if len(thresholds) > 10:  # Limit thresholds for continuous features
                thresholds = np.percentile(X[:, feature_index], [25, 50, 75])
            
            for threshold in thresholds:
                left_mask = X[:, feature_index] < threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                    
                gain = self._information_gain(y, left_mask, right_mask, parent_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _information_gain(self, y, left_mask, right_mask, parent_entropy):
        """
        Calculates the information gain from a potential split.
        
        Args:
            y (ndarray): Target values
            left_mask (ndarray): Boolean mask for left split
            right_mask (ndarray): Boolean mask for right split
            parent_entropy (float): Entropy of the parent node
            
        Returns:
            float: Information gain from the split
        """
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        if n_left == 0 or n_right == 0:
            return 0
            
        child_entropy = (n_left * self._entropy(y[left_mask]) + 
                        n_right * self._entropy(y[right_mask])) / (n_left + n_right)
        return parent_entropy - child_entropy

    def _entropy(self, y):
        """
        Calculates the Shannon entropy of a target vector.
        
        Args:
            y (ndarray): Target values
            
        Returns:
            float: Entropy value (in bits)
        """
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts[counts > 0] / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))  # Small epsilon to avoid log(0)

    def _most_common_label(self, y):
        """
        Returns the most common class label in the input vector.
        
        Args:
            y (ndarray): Target values
            
        Returns:
            int: The most frequent class label
        """
        return Counter(y).most_common(1)[0][0] if len(y) > 0 else 0

    def predict(self, X):
        """
        Predicts class labels for samples in X.
        
        Args:
            X (ndarray): Input samples of shape (n_samples, n_features)
            
        Returns:
            ndarray: Predicted class labels
            
        Raises:
            ValueError: If the model hasn't been trained yet
        """
        if self.tree is None:
            raise ValueError("The model must be trained before making predictions")
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x, tree=None):
        """
        Predicts the class label for a single sample by traversing the tree.
        
        Args:
            x (ndarray): A single input sample
            tree (dict, optional): Current node in the tree (used internally during recursion)
            
        Returns:
            int: Predicted class label
        """
        if tree is None:
            tree = self.tree
            
        if isinstance(tree, dict):
            if x[tree["feature_index"]] < tree["threshold"]:
                return self._predict_single(x, tree["left"])
            else:
                return self._predict_single(x, tree["right"])
        return tree

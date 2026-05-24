import numpy as np
from collections import Counter
from ..classification.decision_tree import DecisionTree

class RandomForest:
   
    """
    Random Forest algorithm for binary and multi-class classification.
 
    Description:
        Implements the Random Forest algorithm from scratch using an ensemble of
        DecisionTree classifiers. Each tree is trained on a bootstrap sample of
        the data (sampling with replacement) and considers only a random subset
        of features at each split (feature subsampling). Final predictions are
        determined by majority vote across all trees, which reduces variance and
        improves generalisation over a single decision tree.
 
    Args:
        n_estimators (int): Number of trees in the forest. Defaults to 100.
        max_depth (int, optional): Maximum depth of each tree. Defaults to None.
        min_samples_split (int): Minimum samples required to split a node. Defaults to 2.
        min_samples_leaf (int): Minimum samples required at each leaf. Defaults to 1.
        max_features (int, float, str or None): Number of features to consider at each
            split. Accepted values:
            - int   : exact number of features.
            - float : fraction of total features, must be in (0, 1].
            - "sqrt": int(sqrt(n_features))  [default, recommended for classification].
            - "log2": int(log2(n_features)).
            - None  : all features (no subsampling).
        random_state (int or None): Seed for reproducibility. Defaults to None.
 
    Examples:
        >>> from ifri_mini_ml_lib.classification import RandomForest as rf
        >>> rf = RandomForest(n_estimators=100, max_depth=5, random_state=42)
        >>> rf.fit(X_train, y_train)
        >>> predictions = rf.predict(X_test)
        >>> accuracy = np.mean(predictions == y_test)
    """ 

    def __init__(
        self,
        n_estimators = 100,
        max_depth = None,
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_features = "sqrt",
        random_state = None,
    ):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
 
        self.trees_ = []
        self.n_features_ = None
        self.n_features_per_split_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Build the forest by training each DecisionTree on a bootstrap sample.
 
        Description:
            For each estimator, draws a bootstrap sample from (X, y), selects a
            random subset of features, and fits a DecisionTree on the result.
            The fitted tree and its feature indices are stored together for use
            at prediction time.
 
        Args:
            X (ndarray): Training data of shape (n_samples, n_features).
            y (ndarray): Target labels of shape (n_samples,).
 
        Returns:
            self: The fitted RandomForest instance.
        """
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.random_state)
 
        self.n_features_ = X.shape[1]
        self.n_features_per_split_ = self._resolve_max_features(self.n_features_)
        self.classes_ = np.unique(y)
        self.trees_ = []
 
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y, rng)
            feature_indices = self._sample_features(self.n_features_, rng)
 
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees_.append((tree, feature_indices))
 
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X using majority vote.
 
        Description:
            Each tree in the forest independently predicts a label for every
            sample. The final label is the class that receives the most votes
            across all trees.
 
        Args:
            X (ndarray): Samples of shape (n_samples, n_features).
 
        Returns:
            ndarray: Predicted class labels of shape (n_samples,).
 
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        # Collect predictions from every tree → shape (n_estimators, n_samples)
        all_preds = np.array([
            tree.predict(X[:, feature_indices])
            for tree, feature_indices in self.trees_
        ])
        # Majority vote per sample
        return np.array([
            Counter(all_preds[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
 
        Description:
            For each sample, counts how many trees voted for each class and
            divides by the total number of trees. The resulting probabilities
            reflect the confidence of the ensemble for each class.
 
        Args:
            X (ndarray): Samples of shape (n_samples, n_features).
 
        Returns:
            ndarray: Probability matrix of shape (n_samples, n_classes),
                     columns ordered as in self.classes_.
 
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
 
        vote_counts = np.zeros((n_samples, n_classes))
 
        for tree, feature_indices in self.trees_:
            preds = tree.predict(X[:, feature_indices])
            for sample_idx, label in enumerate(preds):
                vote_counts[sample_idx, class_to_idx[label]] += 1
 
        return vote_counts / self.n_estimators
    

    def _bootstrap_sample(self, X, y, rng ):
        """
        Draw a bootstrap sample (with replacement) from (X, y).
 
        Description:
            Randomly samples n_samples indices with replacement, so some rows
            appear multiple times and others not at all. Each tree in the forest
            receives a different bootstrap sample, which is the core of bagging.
 
        Args:
            X (ndarray): Feature matrix of shape (n_samples, n_features).
            y (ndarray): Label vector of shape (n_samples,).
            rng (Generator): NumPy random generator.
 
        Returns:
            tuple: (X_sample, y_sample) of the same shape as the input.
        """
    
        n_samples = X.shape[0]
        indices = rng.integers(0, n_samples, size=n_samples)
        return X[indices], y[indices]
    


    def _sample_features(self, n_features, rng):
        """
        Randomly select a subset of features for a tree.
 
        Description:
            Draws k feature indices uniformly at random, where k is determined
            by max_features. This ensures each tree only sees a subset of the
            available features, increasing diversity across the ensemble.
 
        Args:
            n_features (int): Total number of features available.
            rng (Generator): NumPy random generator.
 
        Returns:
            ndarray: Sorted array of selected feature indices of length k.
        """
        k = self.n_features_per_split_
        indices = rng.choice(n_features, size=k, replace=False)
        return np.sort(indices)
    

    def _resolve_max_features(self, n_features):
        """Determines the number of features to consider at each split based on the max_features parameter.
        
        Args:
            n_features (int): Total number of features in the dataset.
        Returns:
            int: Number of features to sample at each split.
        """
        mf = self.max_features
 
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if mf is None:
            return n_features
        if isinstance(mf, float):
            if not (0.0 < mf <= 1.0):
                raise ValueError("max_features as a float must be in (0, 1].")
            return max(1, int(mf * n_features))
        if isinstance(mf, int):
            if not (1 <= mf <= n_features):
                raise ValueError(
                    f"max_features={mf} is out of range [1, {n_features}]."
                )
            return mf
 
        raise ValueError(
            f"Unknown max_features value: {mf!r}. "
            "Expected 'sqrt', 'log2', None, an int, or a float in (0, 1]."
        )
    
    def _check_is_fitted(self):
        """
        Check if the model has been fitted by verifying that trees_ is not empty.
 
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.trees_:
            raise ValueError("This RandomForest instance is not fitted yet. ")
    


 



import numpy as np

def c(n):
    """
    Computes the average path length of unsuccessful searches in a BST.

    Description:
        Normalization function used to estimate the average depth of an
        isolation tree. Based on the harmonic approximation from
        Liu et al. (2008).

    Args:
        n (int): Number of points in the sub-sample.

    Returns:
        float: Estimated average path length.
    
    Examples:
        >>> c(256)
        10.244...
        >>> c(1)
        0
    """
    if n <= 1:
        return 0
    elif n == 2:
        return 1
    else:
        H = np.log(n-1) + 0.5772156649
        return 2 * H - ( 2*(n-1) / n)


class Node:
    """
    Represents a node in an iTree.

    Description:
        Can be either an internal node (with a split) or a leaf node.
        Internal nodes store the split axis and threshold along with
        left and right children. Leaf nodes only store the number of
        points they contain.

    Attributes:
        feature (int): Randomly chosen split axis.
        threshold (float): Randomly chosen split value.
        left (Node): Left child (x[feature] < threshold).
        right (Node): Right child (x[feature] >= threshold).
        size (int): Number of points in the leaf node.
        is_leaf (bool): True if the node is a leaf.
    """
    def __init__(self):
        self.feature   = None  
        self.threshold = None  
        self.left      = None  
        self.right     = None  
        self.size      = None  
        self.is_leaf   = False


def build_itree(X, current_depth, height_limit, rng):
    
    """
    Recursively builds an iTree on the subset X.

    Description:
        At each internal node, only the features with a non-zero range
        (i.e. where points are not all identical) are considered as
        candidate split axes. A split axis and a split value are then
        chosen uniformly at random among those candidates. If no such
        feature exists, or if the depth limit is reached, or if the
        subset contains a single point, the node becomes a leaf.

    Args:
        X (np.ndarray): Current node points of shape (n, d).
        current_depth (int): Current depth in the tree.
        height_limit (int): Maximum allowed depth.
        rng (np.random.Generator): Random number generator for reproducibility.

    Returns:
        Node: Root node of the constructed subtree.
    
    Examples:
        >>> rng = np.random.default_rng(42)
        >>> tree = build_itree(X_sample, current_depth=0, height_limit=8, rng=rng)
    """
    
    node = Node()
    n, d  = X.shape

    # ── Compute the range of each feature ──
    ranges      = X.max(axis=0) - X.min(axis=0)   # shape (d,)
    valid_axes  = np.where(ranges > 0)[0]          # indices of splittable axes

    # ── Stopping conditions ──
    if current_depth >= height_limit or n <= 1 or len(valid_axes) == 0:
        node.is_leaf = True
        node.size    = n
        return node

    # ── Choose the split axis among valid axes ──
    axis      = rng.choice(valid_axes)
    axis_min  = X[:, axis].min()
    axis_max  = X[:, axis].max()

    # ── Choose the split value ──
    cut_value        = rng.uniform(axis_min, axis_max)
    node.feature     = axis
    node.threshold   = cut_value

    # ── Partition the space ──
    goes_left   = X[:, axis] < cut_value
    X_left      = X[goes_left]
    X_right     = X[~goes_left]

    # ── Recurse on each subspace ──
    node.left  = build_itree(X_left,  current_depth + 1, height_limit, rng)
    node.right = build_itree(X_right, current_depth + 1, height_limit, rng)

    return node
    

def path_length(x, node, current_depth):
    """
    Computes the path length h(x) of a point in an iTree.

    Description:
        Recursively traverses the tree following the split conditions
        until reaching a leaf node. Returns the depth reached plus
        an adjustment term c(size) to account for the remaining
        estimated depth.

    Args:
        x (np.ndarray): Single input point of shape (d,).
        node (Node): Current node in the traversal.
        current_depth (int): Current depth in the traversal.

    Returns:
        float: Estimated path length for the point x.
    """
    
    if node.is_leaf:
        return current_depth + c(node.size)

    # ── Internal node: follow the split ──
    if x[node.feature] < node.threshold:
        return path_length(x, node.left,  current_depth + 1)
    else:
        return path_length(x, node.right, current_depth + 1)


class IsolationForest:
    
    """
    Isolation Forest algorithm for anomaly detection.

    Description:
        Implements the Isolation Forest algorithm from Liu et al. (2008).
        Anomalies are isolated faster than normal points, resulting in
        shorter average path lengths across the ensemble of iTrees.
        A higher anomaly score (close to 1) indicates an anomaly,
        while a score close to 0.5 indicates a normal point.

    Args:
        n_trees (int, optional): Number of isolation trees. Default is 100.
        sample_size (int, optional): Sub-sample size used to build each tree. Default is 256.
        contamination (float, optional): Expected proportion of anomalies in the data. Default is 0.1.
        random_state (int or None, optional): Seed for the random number generator.
            Set to an integer for fully reproducible results. Default is None.

    Attributes:
        trees (list): List of fitted iTrees (Node roots).
        c_sample_size (float): Normalization constant c(sample_size).
        threshold (float): Decision threshold computed during fit.
        rng (np.random.Generator): Internal random number generator.

    Examples:
        >>> model = IsolationForest(n_trees=100, sample_size=256, contamination=0.1, random_state=42)
        >>> model.fit(X_train)
        >>> scores = model.anomaly_score(X_test)
        >>> labels = model.predict(X_test)
    """
    
    def __init__(self, n_trees=100, sample_size=256, contamination=0.1, random_state=None):
        self.n_trees       = n_trees
        self.sample_size   = sample_size
        self.contamination = contamination
        self.random_state  = random_state
        self.trees         = []
        self.rng           = np.random.default_rng(random_state)
    
    def fit(self, X):
        
        """
        Builds the isolation forest on the training data.

        Description:
            Constructs n_trees isolation trees, each trained on a random
            sub-sample of size sample_size. Also computes the decision
            threshold based on the contamination rate.

            Note on the threshold: it is calibrated on the training data,
            which may include anomalies. If the true contamination rate is
            unknown or uncertain, prefer using anomaly_score() directly and
            selecting the threshold via external validation (e.g. a labeled
            validation set or a precision-recall curve).

        Args:
            X (array-like): Training data of shape (n_samples, n_features).

        Returns:
            self: The fitted IsolationForest instance.
        """
        
        self.trees = [] 
        
        X = np.array(X)
        n = X.shape[0]

        height_limit       = int(np.ceil(np.log2(self.sample_size)))
        self.c_sample_size = c(self.sample_size)

        sample_size = min(self.sample_size, n)  
        

        for _ in range(self.n_trees):
            sample_idx = self.rng.choice(n, size=sample_size, replace=False)
            X_sample   = X[sample_idx]
            tree       = build_itree(X_sample, current_depth=0, height_limit=height_limit, rng=self.rng)
            self.trees.append(tree)
        
        scores         = self.anomaly_score(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        return self
        
        
    def anomaly_score(self, X):
        
        """
        Computes the anomaly score for each point in X.

        Description:
            For each point, computes the path length across all trees and
            collects the results in a matrix of shape (n_samples, n_trees).
            The average path length E[h(x)] is then computed across trees
            in a single NumPy operation. The final score is normalized using
            c(sample_size) so that it lies in [0, 1].
            A score close to 1 indicates an anomaly, close to 0.5 indicates
            a normal point.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Anomaly scores of shape (n_samples,), values in [0, 1].
        """
        
        X = np.array(X)

        # Depth matrix: shape (n_samples, n_trees)
        all_depths = np.array(
            [[path_length(x, tree, current_depth=0) for x in X]
            for tree in self.trees]
        ).T  # → (n_samples, n_trees)

        # E[h(x)]: average path length across all trees
        E_h = all_depths.mean(axis=1)   # shape (n_samples,)

        # Final score: s(x, ψ) = 2^( -E[h(x)] / c(ψ) )
        return 2 ** (-E_h / self.c_sample_size)
    
    def predict(self, X):
        """
        Classifies each point as anomaly (1) or normal (0).

        Description:
            Computes anomaly scores and compares them against the threshold
            learned during fit. Points with scores above the threshold
            are classified as anomalies.

        Args:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Binary predictions of shape (n_samples,).
                        1 = anomaly, 0 = normal.
        """
        
        scores = self.anomaly_score(X)
        return (scores >= self.threshold).astype(int)
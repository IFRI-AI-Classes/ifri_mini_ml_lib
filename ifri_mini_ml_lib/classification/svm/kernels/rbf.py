"""RBF kernel helpers and a small SMO-based SVM implementation.

This module exposes a vectorized RBF kernel, a compact SMO optimizer,
and two simple classifiers built on top of them.
"""

import numpy as np

"""
def rbf_kernel(x, y, gamma = 1.0) :
    
    
    Computes the RBF (Radial Basis Function) kernel between
    two vectors x and y
    
    
    
    # Compute the difference between the two vectors
    difference = x - y
    
    
    # Compute the squared distance
    squared_distance = np.sum(difference ** 2)
    
    
    # Apply the exponential RBF formula
    rbf_value = np.exp( -gamma * squared_distance )
    
    
    return rbf_value




def rbf_gram_matrix ( X1, X2, gamma = 1.0 ) :
    
    
    Computes the RBF gram matrix between two datasets X1 and X2
    
    
    
    
    # Number of samples (rows) in X1 and X2
    n_samples_1 = X1.shape[0]
    n_samples_2 = X2.shape[0]
    
    
    # Create an empty matrix filled with zeros
    gram_matrix = np.zeros((n_samples_1, n_samples_2))
    
    
    for i in range (n_samples_1) :
        for j in range (n_samples_2) :
            gram_matrix[i, j] = rbf_kernel (X1[i], X2[j], gamma)
            
            
    
    return gram_matrix
    
"""

# ─── Fonctions indépendantes ───────────────────────────────

def rbf_kernel(X, Y, gamma=1.0):
    """Compute the pairwise RBF kernel matrix between two sample sets.

    Args:
        X (np.ndarray): First matrix of shape (n_samples_X, n_features).
        Y (np.ndarray): Second matrix of shape (n_samples_Y, n_features).
        gamma (float): RBF width parameter. Larger values make the kernel
            decay faster with distance.

    Returns:
        np.ndarray: Kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    diff = X[:, np.newaxis, :] - Y  # (n,m,d)
    sq_distances = np.sum(diff**2, axis=2)  # (n,m)
    return np.exp(-gamma * sq_distances)


def smo(K, y, C, tol=1e-3, max_iter=100):
    """Solve the SVM dual problem with a basic SMO loop.

    Args:
        K (np.ndarray): Gram matrix of shape (n_samples, n_samples).
        y (np.ndarray): Binary labels encoded as -1 and +1.
        C (float): Regularization strength.
        tol (float): KKT violation tolerance.
        max_iter (int): Maximum number of outer iterations.

    Returns:
        tuple[np.ndarray, float]: The optimized alpha coefficients and bias.
    """
    n = len(y)
    alphas = np.zeros(n)
    b = 0
    
    for _ in range(max_iter):
        errors = (alphas * y) @ K + b - y
        
        vi = y * errors
        violations = np.concatenate([
            np.where((vi < -tol) & (alphas < C))[0],
            np.where((vi > tol) & (alphas > 0))[0]
        ])
        
        if len(violations) == 0:
            break
            
        i = violations[np.argmax(np.abs(vi[violations]))]
        
        diff_errors = np.abs(errors[i] - errors)
        diff_errors[i] = 0
        j = np.argmax(diff_errors)
        
        eta = K[i,i] + K[j,j] - 2*K[i,j]
        if eta <= 0:
            continue
            
        if y[i] == y[j]:
            L = max(0, alphas[i] + alphas[j] - C)
            H = min(C, alphas[i] + alphas[j])
        else:
            L = max(0, alphas[j] - alphas[i])
            H = min(C, C + alphas[j] - alphas[i])
            
        if L == H:
            continue
            
        alpha_j_new = alphas[j] + y[j] * (errors[i] - errors[j]) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)
        alpha_i_new = alphas[i] + y[i] * y[j] * (alphas[j] - alpha_j_new)
        
        b1 = b - errors[i] - y[i]*(alpha_i_new - alphas[i])*K[i,i] - y[j]*(alpha_j_new - alphas[j])*K[i,j]
        b2 = b - errors[j] - y[i]*(alpha_i_new - alphas[i])*K[i,j] - y[j]*(alpha_j_new - alphas[j])*K[j,j]
        
        if 0 < alpha_i_new < C:
            b = b1
        elif 0 < alpha_j_new < C:
            b = b2
        else:
            b = (b1 + b2) / 2
            
        alphas[i] = alpha_i_new
        alphas[j] = alpha_j_new
    
    return alphas, b


# ─── Classe ───────────────────────────────────────────────

class SVMClassifier:
    """Binary SVM classifier trained with an RBF kernel and SMO."""

    def __init__(self, C=1.0, gamma=1.0):
        """Initialize the classifier.

        Args:
            C (float): Regularization parameter.
            gamma (float): RBF kernel parameter.
        """
        self.C = C
        self.gamma = gamma
        self.support_vectors = None
        self.support_alphas = None
        self.support_labels = None
        self.b = None

    def fit(self, X, y):
        """Fit the classifier and keep only support vectors."""
        K = rbf_kernel(X, X, self.gamma)
        alphas, b = smo(K, y, self.C)
        support_indices = np.where(alphas > 0)[0]
        self.support_vectors = X[support_indices]
        self.support_alphas = alphas[support_indices]
        self.support_labels = y[support_indices]
        self.b = b

    def predict(self, X):
        """Predict class labels for a batch of samples."""
        K = rbf_kernel(self.support_vectors, X, self.gamma)
        scores = (self.support_alphas * self.support_labels) @ K + self.b
        return np.sign(scores)


class SVMClassifierOvO:
    """One-vs-one multi-class wrapper around the binary RBF SVM."""

    def __init__(self, C=1.0, gamma=1.0):
        """Initialize the multi-class wrapper.

        Args:
            C (float): Regularization parameter for each binary classifier.
            gamma (float): RBF kernel parameter.
        """
        self.C = C
        self.gamma = gamma
        self.classifiers = {}
    
    def fit(self, X, y):
        """Train one binary classifier for each pair of classes."""
        from itertools import combinations
        classes = np.unique(y)
        pairs = list(combinations(classes, 2))
        
        for (c1, c2) in pairs:
            mask = (y == c1) | (y == c2)
            X_pair = X[mask]
            y_binary = np.where(y[mask] == c1, 1, -1)
            
            clf = SVMClassifier(C=self.C, gamma=self.gamma)
            clf.fit(X_pair, y_binary)
            self.classifiers[(c1, c2)] = clf
    
    def predict_one(self, x):
        """Predict a single sample by majority vote."""
        votes = {}
        for (c1, c2), clf in self.classifiers.items():
            prediction = clf.predict(x.reshape(1, -1))[0]
            if prediction == 1:
                votes[c1] = votes.get(c1, 0) + 1
            else:
                votes[c2] = votes.get(c2, 0) + 1
        return max(votes, key=votes.get)
    
    def predict(self, X):
        """Predict class labels for a batch of samples."""
        return np.array([self.predict_one(x) for x in X])
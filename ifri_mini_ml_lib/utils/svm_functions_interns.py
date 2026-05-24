import numpy as np
from itertools import combinations

def rbf_kernel(X, Y, gamma):
    """
    Computes the RBF kernel matrix between two sets of points.

    Args:
        X (np.ndarray of shape (n, d)): First set of points.
        Y (np.ndarray of shape (m, d)): Second set of points.

    Returns:
        np.ndarray of shape (n, m): Kernel matrix.
    """
    # diff[i,j,:] = X[i] - Y[j]  →  shape (n, m, d)
    diff = X[:, np.newaxis, :] - Y
    # squared distances between each pair of points → shape (n, m)
    sq_distances = np.sum(diff ** 2, axis=2)
    return np.exp(-gamma * sq_distances)

def smo(K, y, C, tol, max_iter):
    """
    Solves the dual SVM problem via the SMO algorithm.

    Description:
        SMO (Sequential Minimal Optimization) optimizes Lagrange
        multipliers (alphas) by updating them pairwise at each
        iteration. 
    
    Args:
        K (np.ndarray of shape (n, n)): Gram matrix (pre-computed kernel).
        y (np.ndarray of shape (n,))  : Labels in {-1, +1}.

    Returns:
        tuple (alphas, b):
            - alphas (np.ndarray): Optimal Lagrange multipliers.
            - b (float)          : Optimal bias.
    """
    n = len(y)
    alphas = np.zeros(n)
    b = 0.0

    for _ in range(max_iter):
        # Prediction errors for all examples
        errors = (alphas * y) @ K + b - y

        # Identify KKT condition violations
        vi = y * errors
        violations = np.concatenate([
            np.where((vi < -tol) & (alphas < C))[0],
            np.where((vi > tol) & (alphas > 0))[0]
        ])

        # Convergence: no more violations
        if len(violations) == 0:
            break

        # Select the worst violator as first point i
        i = violations[np.argmax(np.abs(vi[violations]))]

        # Select j with the most different error from i
        diff_errors = np.abs(errors[i] - errors)
        diff_errors[i] = 0
        j = np.argmax(diff_errors)

        # Compute eta (curvature of the objective)
        eta = K[i, i] + K[j, j] - 2 * K[i, j]
        if eta <= 0:
            continue

        # Compute bounds L and H for alpha_j
        if y[i] == y[j]:
            L = max(0, alphas[i] + alphas[j] - C)
            H = min(C, alphas[i] + alphas[j])
        else:
            L = max(0, alphas[j] - alphas[i])
            H = min(C, C + alphas[j] - alphas[i])

        if L == H:
            continue

        # Update alpha_j then alpha_i
        alpha_j_new = alphas[j] + y[j] * (errors[i] - errors[j]) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)
        alpha_i_new = alphas[i] + y[i] * y[j] * (alphas[j] - alpha_j_new)

        # Update bias b
        b1 = (b - errors[i]
                - y[i] * (alpha_i_new - alphas[i]) * K[i, i]
                - y[j] * (alpha_j_new - alphas[j]) * K[i, j])
        b2 = (b - errors[j]
                - y[i] * (alpha_i_new - alphas[i]) * K[i, j]
                - y[j] * (alpha_j_new - alphas[j]) * K[j, j])

        if 0 < alpha_i_new < C:
            b = b1
        elif 0 < alpha_j_new < C:
            b = b2
        else:
            b = (b1 + b2) / 2.0

        alphas[i] = alpha_i_new
        alphas[j] = alpha_j_new

    return alphas, b

def hinge_loss(X, y, w, b, lam):
    """
    Computes the regularized hinge loss function (Pegasos objective).

    Args:
        X   (np.ndarray): Training data.
        y   (np.ndarray): Labels in {-1, +1}.
        w   (np.ndarray): Current weight vector.
        b   (float)     : Current bias.
        lam (float)     : Regularization parameter λ = 1/C.

    Returns:
        float: Value of the objective function.
    """
    margins = y * (X.dot(w) + b)
    hinge = np.maximum(0, 1 - margins)
    regularization = (lam / 2.0) * np.dot(w, w)
    return float(regularization + np.mean(hinge))

def pegasos(X, y, C, max_iter, tol, random_state=None):
    """
    Executes the Pegasos algorithm to find optimal w* and b*.

    Args:
        X (np.ndarray of shape (n_samples, n_features)): Training data.
        y (np.ndarray of shape (n_samples,))           : Labels in {-1, +1}.

    Returns:
        tuple (w, b, loss_history):
            - w (np.ndarray of shape (n_features,)): optimal weights
            - b (float)                            : optimal bias
            - loss_history (list)                  : loss at each epoch

    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape
    lam = 1.0 / C          # λ = 1/C
    prev_loss = float('inf')

    w = np.zeros(n_features)
    b = 0.0
    loss_history = []

    for epoch in range(1, max_iter + 1):
        # Random shuffling of examples at each epoch (SGD)
        indices = np.random.permutation(n_samples)

        for t, i in enumerate(indices, start=1):
            # Decaying learning rate
            t_global = (epoch - 1) * n_samples + t
            eta = 1.0 / (lam * t_global)

            # Compute margin for current example
            margin = y[i] * (np.dot(w, X[i]) + b)

            # Update based on margin
            if margin < 1:
                # Misclassified example → correct w and b
                w = (1 - eta * lam) * w + eta * y[i] * X[i]
                b += eta * y[i]
            else:
                # Well-classified example → regularization only
                w = (1 - eta * lam) * w

            # Projection: bring w back into ball of radius 1/√λ
            norm_w = np.linalg.norm(w)
            proj_radius = 1.0 / np.sqrt(lam)
            if norm_w > proj_radius:
                w *= proj_radius / norm_w

        # Compute loss at end of each epoch
        epoch_loss = hinge_loss(X, y, w, b, lam)
        loss_history.append(epoch_loss)

        # Stopping criterion: loss convergence
        if abs(prev_loss - epoch_loss) < tol:
            break
        prev_loss = epoch_loss

    return w, b, loss_history

def fit_linear(X, y, classes, C, max_iter, tol, random_state):
    """
    Trains the linear SVM via Pegasos.

    Description:
        Handles binary and multiclass classification automatically:
        - Binary     : single call to Pegasos
        - Multiclass : One-vs-Rest (OvR) strategy — one binary
                        classifier per class

    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Labels (converted to {-1, +1} internally).
    
    Returns:
        tuple: (w, b, loss_history) or (w_list, b_list, loss_history_list) for multiclass
    """
    if len(classes) == 2:
        # Binary case
        y_binary = np.where(y == classes[1], 1, -1).astype(float)
        w, b, loss_history = pegasos(X, y_binary, C, max_iter, tol, random_state)
        return w, b, loss_history
    else:
        # Multiclass case: One-vs-Rest
        w_list = []
        b_list = []
        loss_history_list = []
        for cls in classes:
            y_binary = np.where(y == cls, 1, -1).astype(float)
            w, b, loss = pegasos(X, y_binary, C, max_iter, tol, random_state)
            w_list.append(w)
            b_list.append(b)
            loss_history_list.append(loss)
        return w_list, b_list, loss_history_list

def fit_rbf(X, y, classes, C, gamma, tol, max_iter):
    """
    Trains the RBF SVM via SMO.

    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Original labels.
    
    Returns:
        tuple: (support_vectors, support_alphas, support_labels, b) or 
               (rbf_classifiers) for multiclass
    """
    if len(classes) == 2:
        # Binary case
        y_binary = np.where(y == classes[1], 1, -1).astype(float)
        K = rbf_kernel(X, X, gamma)
        alphas, b = smo(K, y_binary, C, tol, max_iter)
        sv_idx = alphas > 0
        support_vectors = X[sv_idx]
        support_alphas = alphas[sv_idx]
        support_labels = y_binary[sv_idx]
        return support_vectors, support_alphas, support_labels, b
    else:
        # Multiclass case: One-vs-One
        rbf_classifiers = {}
        pairs = list(combinations(classes, 2))
        for (c1, c2) in pairs:
            mask = (y == c1) | (y == c2)
            X_pair = X[mask]
            y_binary = np.where(y[mask] == c1, 1, -1).astype(float)
            K = rbf_kernel(X_pair, X_pair, gamma)
            alphas, b = smo(K, y_binary, C, tol, max_iter)
            sv_idx = alphas > 0
            rbf_classifiers[(c1, c2)] = {
                'sv': X_pair[sv_idx],
                'alphas': alphas[sv_idx],
                'labels': y_binary[sv_idx],
                'b': b
            }
        return rbf_classifiers

def predict_linear(X, w_, b_, classes_):
    """
    Predicts labels with the linear kernel.

    Args:
        X (np.ndarray): Test data.

    Returns:
        np.ndarray: Predicted labels.
    """
    if len(classes_) == 2:
        scores = X.dot(w_) + b_
        return np.where(scores >= 0, classes_[1], classes_[0])
    else:
        # OvR: class with maximum score
        all_scores = np.column_stack([
            X.dot(w) + b for w, b in zip(w_, b_)
        ])
        return classes_[np.argmax(all_scores, axis=1)]
    
def predict_rbf(X, support_vectors_, support_alphas_, support_labels_, b_, classes_, rbf_classifiers_, gamma):
    """
    Predicts labels with the RBF kernel.

    Args:
        X (np.ndarray): Test data.
        support_vectors_ (np.ndarray or None): Support vectors from training (binary only).
        support_alphas_ (np.ndarray or None): Lagrange multipliers (binary only).
        support_labels_ (np.ndarray or None): Labels of support vectors (binary only).
        b_ (float or None): Bias term (binary only).
        classes_ (np.ndarray): Unique classes.
        rbf_classifiers_ (dict): OvO classifiers for multiclass.
        gamma (float): RBF kernel parameter.

    Returns:
        np.ndarray: Predicted labels.
    """
    if support_vectors_ is not None:
        # Binary case
        K = rbf_kernel(support_vectors_, X, gamma)
        scores = (support_alphas_ * support_labels_) @ K + b_
        preds = np.sign(scores)
        return np.where(preds >= 0, classes_[1], classes_[0])
    else:
        # OvO: majority vote (multiclass)
        votes = np.zeros((len(X), len(classes_)))
        class_to_idx = {cls: i for i, cls in enumerate(classes_)}
        for (c1, c2), clf in rbf_classifiers_.items():
            K = rbf_kernel(clf['sv'], X, gamma)
            scores = (clf['alphas'] * clf['labels']) @ K + clf['b']
            for k, score in enumerate(scores):
                if score >= 0:
                    votes[k, class_to_idx[c1]] += 1
                else:
                    votes[k, class_to_idx[c2]] += 1
        return classes_[np.argmax(votes, axis=1)]
    

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from ifri_mini_ml_lib.classification import KNN

# 1. Generate synthetic data
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42, cluster_std=2.0)

def plot_knn(k, metric, library):
    # 2. Setup the KNN Model
    if library == 'sklearn':
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
    else:  # ifri_mini_ml_lib
        clf = KNN(k=k, task='classification')

    clf.fit(X, y)

    # 3. Create a mesh grid for decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict across the entire mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50, cmap='viridis')

    metric_info = f", metric='{metric}'" if library == 'sklearn' else ""
    plt.title(f"KNN Decision Boundaries ({library}) (k={k}{metric_info})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

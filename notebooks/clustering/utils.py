import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris


def create_kmeans_demo_data(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42,
):
    """Create the synthetic dataset used in the K-Means documentation notebook."""

    return make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )


def load_hierarchical_iris_data():
    """Load the Iris dataset used in the hierarchical clustering notebook."""

    iris = load_iris()
    return {
        "data": iris.data,
        "target": iris.target,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names,
    }


def plot_dataset(X, title="Original Dataset", xlabel="Feature 1", ylabel="Feature 2"):
    """Plot a 2-D dataset without labels."""

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[:, 0], X[:, 1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_cluster_assignments(
    X,
    labels,
    title,
    centroids=None,
    xlabel="Feature 1",
    ylabel="Feature 2",
    figsize=(7, 5),
):
    """Plot 2-D clusters using a consistent notebook-friendly style."""

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    cmap = plt.colormaps.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=figsize)
    for index, label in enumerate(unique_labels):
        points = X[labels == label]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=cmap(index),
            label=f"Cluster {label}",
            s=50,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.4,
        )

    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=200,
            c="black",
            linewidths=2,
            label="Centroids",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_hierarchical_feature_comparison(
    X_plot,
    labels,
    y_true,
    feature_names,
    target_names,
    linkage,
    k,
    silhouette_score,
):
    """Plot the learned clustering next to the reference Iris labels."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.colormaps.get_cmap("tab10")

    ax = axes[0]
    for index in range(k):
        points = X_plot[labels == index]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=cmap(index),
            label=f"Cluster {index}",
            s=40,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.4,
        )
    ax.set_xlabel(feature_names[0], fontsize=11)
    ax.set_ylabel(feature_names[1], fontsize=11)
    ax.set_title(
        f"Clustering hiérarchique\nlinkage={linkage}, k={k} | Silhouette = {silhouette_score:.3f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    colors_true = ["#E74C3C", "#2ECC71", "#3498DB"]
    for index, name in enumerate(target_names):
        points = X_plot[y_true == index]
        ax2.scatter(
            points[:, 0],
            points[:, 1],
            color=colors_true[index],
            label=name,
            s=40,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.4,
        )
    ax2.set_xlabel(feature_names[0], fontsize=11)
    ax2.set_ylabel(feature_names[1], fontsize=11)
    ax2.set_title("Vérité terrain (espèces réelles)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_hierarchical_dendrogram_demo(hierarchical_model, labels=None):
    """Display the dendrogram for a fitted hierarchical clustering model."""

    plt.figure(figsize=(14, 5))
    hierarchical_model.plot_dendrogram(labels=labels)


def plot_hierarchical_method_comparison(results):
    """Plot agglomerative vs divisive clustering results side by side."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.colormaps.get_cmap("tab10")

    for ax, result in zip(axes, results):
        labels = np.asarray(result["labels"])
        points = result["X"]
        for index in np.unique(labels):
            cluster_points = points[labels == index]
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=cmap(int(index)),
                label=f"Cluster {index}",
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel(result["xlabel"])
        ax.set_ylabel(result["ylabel"])
        ax.set_title(
            f"{result['title']}\nSilhouette = {result['silhouette']:.3f} | Temps = {result['elapsed']:.3f}s",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Clustering hiérarchique — Iris (50 points, k=3, complete linkage)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()
    return fig, axes


def plot_hierarchical_linkage_comparison(X, results):
    """Plot the effect of different linkage criteria on the same dataset."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, result in zip(axes, results):
        labels = np.asarray(result["labels"])
        unique_labels = np.unique(labels)
        for index in unique_labels:
            points = X[labels == index]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                label=f"Cluster {index}",
                s=50,
                alpha=0.8,
            )

        ax.set_title(
            f'linkage="{result["linkage"]}"\nSilhouette = {result["silhouette"]:.3f}',
            fontsize=11,
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Impact du critère de linkage sur le clustering",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.show()
    return fig, axes

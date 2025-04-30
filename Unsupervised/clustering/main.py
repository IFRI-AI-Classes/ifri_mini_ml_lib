import sys
sys.path.append('./clustering')
from clustering.algo.kmeans import KMeans
from clustering.algo.dbscan import DBSCAN
from clustering.algo.hierarchical import HierarchicalClustering
from clustering.utils.utils import euclidean_distance
from clustering.metrics.inertia import calculate_inertia
from clustering.metrics.silhouette import calculate_silhouette
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

def main():
    # Load data from CSV
    try:
        data = pd.read_csv('./data/data.csv')
        X = data.values  # Convert data to NumPy array
    except FileNotFoundError:
        print("Error : Le fichier data.csv was not found. Make sure it is in the 'data' folder.")
        return

    # Ask to User to choose the algorithm
    print("Choose algorithm to use :")
    print("1. KMeans")
    print("2. DBSCAN")
    print("3. Hierarchical")
    choix = input("Enter the number of your votre choice (1, 2 ou 3) : ")

    if choix == '1':
        # Use algorithm KMeans
        n_clusters = int(input("Enter the number of clusters wished : "))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        print("Labels of clusters (KMeans) :", labels)

        # Calculate and print inertia score
        inertia = calculate_inertia(X, labels, kmeans.centroids)
        print(f"Inertia: {inertia}")

        # Calculate and print silhouette score
        silhouette = calculate_silhouette(X, labels)
        print("Silhouette: {silhouette}")

        # Visualize clusters (if data are 2D)
        if X.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            for i in range(kmeans.n_clusters):
                plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i}')
            plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='x', s=200, color='black', label='Centroids')
            plt.title('Clusters KMeans')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()
        else:
            print("Data aren't 2D, the visualization isn't possible.")

    elif choix == '2':
        # Use algorithm DBSCAN
        eps = float(input("Enter the value of eps (ray) : "))
        min_samples = int(input("Enter the minimum number of samples : "))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        print("Labels of clusters (DBSCAN) :", labels)

        # Calculate and print silhouette score
        silhouette = calculate_silhouette(X, labels)
        print(f"Silhouette: {silhouette}")

        # Visualize clusters 
        if X.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                xy = X[class_member_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=10)

            plt.title('Clusters DBSCAN')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()
        else:
            print("Data aren't 2D. The visualization isn't possible.")
    elif choix == '3':
        # Use algorithm Hierarchical
        n_clusters = int(input("Enter the number of clusters whised : "))
        linkage = input("Enter the linkage criteron ('single', 'complete', 'average') : ")
        method = input("Enter method ('agglomerative', 'divisive') : ")

        hierarchical = HierarchicalClustering(n_clusters=n_clusters, linkage=linkage, method=method)
        labels = hierarchical.fit_predict(X)

        print("Labels of clusters (Hierarchical) :", labels)

        # Calculate and print silhouette score
        silhouette = calculate_silhouette(X, labels)
        print(f"Silhouette: {silhouette}")

        # Print dendrogram (if method is agglomerative)
        if method == 'agglomerative':
            hierarchical.plot_dendrogram(X)

        # Visualize clusters (if data are 2D)
        if X.shape[1] == 2:
            plt.figure(figsize=(8, 6))
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                class_member_mask = (labels == k)
                xy = X[class_member_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=10)

            plt.title('Clusters Hierarchical')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()
        else:
            print("Data aren't 2D, the visualisation isn't possible.")
    else:
        print("Choice invalid. Please choice 1, 2 ou 3.")

if __name__ == "__main__":
    main()

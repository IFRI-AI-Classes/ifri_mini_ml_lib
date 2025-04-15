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
    # Charger les données à partir du fichier CSV
    try:
        data = pd.read_csv('./data/data.csv')
        X = data.values  # Convertir les données en un tableau NumPy
    except FileNotFoundError:
        print("Erreur : Le fichier data.csv n'a pas été trouvé. Assurez-vous qu'il se trouve dans le dossier 'data'.")
        return

    # Demander à l'utilisateur de choisir l'algorithme
    print("Choisissez l'algorithme à utiliser :")
    print("1. KMeans")
    print("2. DBSCAN")
    print("3. Hierarchical")
    choix = input("Entrez le numéro de votre choix (1, 2 ou 3) : ")

    if choix == '1':
        # Utiliser l'algorithme KMeans
        n_clusters = int(input("Entrez le nombre de clusters souhaité : "))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        print("Labels des clusters (KMeans) :", labels)

        # Calculer et afficher l'inertie
        inertia = calculate_inertia(X, labels, kmeans.centroids)
        print(f"Inertie: {inertia}")

        # Calculer et afficher le score silhouette
        silhouette = calculate_silhouette(X, labels)
        print(f"Silhouette: {silhouette}")

        # Visualiser les clusters (si les données sont en 2D)
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
            print("Les données ne sont pas en 2D, la visualisation n'est pas possible.")

    elif choix == '2':
        # Utiliser l'algorithme DBSCAN
        eps = float(input("Entrez la valeur de eps (rayon) : "))
        min_samples = int(input("Entrez le nombre minimal de samples : "))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        print("Labels des clusters (DBSCAN) :", labels)

        # Calculer et afficher le score silhouette
        silhouette = calculate_silhouette(X, labels)
        print(f"Silhouette: {silhouette}")

        # Visualiser les clusters (si les données sont en 2D)
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
            print("Les données ne sont pas en 2D, la visualisation n'est pas possible.")
    elif choix == '3':
        # Utiliser l'algorithme Hierarchical
        n_clusters = int(input("Entrez le nombre de clusters souhaité : "))
        linkage = input("Entrez le critère de linkage ('single', 'complete', 'average') : ")
        method = input("Entrez la méthode ('agglomerative', 'divisive') : ")

        hierarchical = HierarchicalClustering(n_clusters=n_clusters, linkage=linkage, method=method)
        labels = hierarchical.fit_predict(X)

        print("Labels des clusters (Hierarchical) :", labels)

        # Calculer et afficher le score silhouette
        silhouette = calculate_silhouette(X, labels)
        print(f"Silhouette: {silhouette}")

        # Afficher le dendrogramme (si la méthode est agglomérative)
        if method == 'agglomerative':
            hierarchical.plot_dendrogram(X)

        # Visualiser les clusters (si les données sont en 2D)
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
            print("Les données ne sont pas en 2D, la visualisation n'est pas possible.")
    else:
        print("Choix invalide. Veuillez choisir 1, 2 ou 3.")

if __name__ == "__main__":
    main()

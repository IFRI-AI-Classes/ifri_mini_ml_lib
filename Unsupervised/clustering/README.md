# mini-ml-lib : Module de Clustering

Ce module fait partie du projet `mini-ml-lib`, qui vise à créer une bibliothèque d'apprentissage automatique simplifiée à partir de zéro. Ce module se concentre sur l'implémentation d'algorithmes de clustering.

## Objectifs
*   Implémenter les algorithmes de clustering : k-means, DBSCAN et clustering hiérarchique.
*   Fournir des métriques d'évaluation pour les clusters : score silhouette et inertie.
*   Offrir une option de visualisation pour les résultats du clustering.

## Installation
1.  Clonez le dépôt `mini-ml-lib`.
2.  Accédez au répertoire du projet.
3.  Installez les dépendances :

    ```
    pip install -r requirements.txt
    ```

## Structure du code
.
├── clustering/
│ ├── algos/
│ │ ├── kmeans.py # Implémentation de k-means
│ │ ├── dbscan.py # Implémentation de DBSCAN
│ │ └── hierarchical.py # Implémentation du clustering hiérarchique
│ ├── metrics/
│ │ ├── silhouette.py # Métrique score silhouette
│ │ └── inertia.py # Métrique inertie
| |__ utils/
|    |__utils.py # contient les fonctions 
│ └── init.py
├── data/
│ └── data.csv # Jeu de données pour les tests
├── main.py # Script principal
├── requirements.txt # Dépendances du projet
└── README.md # Ce fichier


## Utilisation Kmeans
1.  Importez la classe `KMeans` depuis le module `clustering.algos.kmeans`.
2.  Chargez vos données à partir d'un fichier CSV à l'aide de pandas.
3.  Instanciez la classe `KMeans` avec le nombre de clusters souhaité.
4.  Appelez la méthode `fit_predict` pour effectuer le clustering.
5.  Visualisez les résultats à l'aide de la méthode `plot_clusters` (si les données sont en 2D).

import pandas as pd
from clustering.algos.kmeans import KMeans
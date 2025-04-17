import numpy as np
import matplotlib.pyplot as plt

class ACP:
    """
    Implémentation de l'Analyse en Composantes Principales (PCA) pour la réduction de dimensionnalité.

    Attributes:
        n_component (int): Nombre de composantes principales à conserver
        mean (np.ndarray): Moyenne des features calculée pendant l'entraînement
        std (np.ndarray): Écart-type des features calculé pendant l'entraînement
        cov (np.ndarray): Matrice de covariance calculée
        eigen_values (np.ndarray): Valeurs propres triées
        eigen_vectors (np.ndarray): Vecteurs propres triés
        components (np.ndarray): Composantes principales sélectionnées
    """


    def __init__(self, n_component: int):
        """
        Initialise le modèle ACP.

        Args:
            n_component (int): Nombre de composantes principales à conserver

        Example:
            >>> model = ACP(n_component=2)
        """

        self.n_component = n_component
        self.mean = None
        self.std = None
        self.cov = None
        self.eigen_values = None
        self.eigen_vectors = None
        self.components = None


    def fit(self, X: np.ndarray) -> 'ACP':
        """
        Apprentissage du modèle (calcul des statistiques et composantes).

        Args:
            X (np.ndarray): Matrice 2D de données (shape: [n_samples, n_features])

        Returns:
            ACP: Instance entraînée

        Example:
            >>> data = np.array([[1, 2], [3, 4], [5, 6]])
            >>> model.fit(data)
        """

        # Centrage et normalisation
        self.mean = np.mean(X, axis =  0)
        self.std = np.std(X, axis = 0)
        X = (X - self.mean) / (self.std + 1e-10)
        
        # Calcul de la matrice de covarience
        self.cov = np.cov(X, rowvar=False) # rowvar=False pour considérer les colonnes comme variables

        # Decomposition en valeur propre et vecteur propre
        self.eigen_values, eigen_vectors = np.linalg.eig(self.cov)
        
        # Trie des valeurs propres et vecteurs propres par ordre decroissant
        indices = np.argsort(self.eigen_values)[::-1]
        self.eigen_values = self.eigen_values[indices]
        self.eigen_vectors = eigen_vectors[:, indices]

        # Sélectionner les n premiers vecteurs propres
        self.components = self.eigen_vectors[:, :self.n_component]

        return self
    

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projection des données dans l'espace réduit.

        Args:
            X (np.ndarray): Données à transformer (shape: [n_samples, n_features])

        Returns:
            np.ndarray: Données projetées (shape: [n_samples, n_component])

        Example:
            >>> transformed_data = model.transform(data)
            >>> print(transformed_data.shape)
            (3, 2)
        """

        X = (X - self.mean) / (self.std + 1e-10)
        return np.dot(X, self.components)
    
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apprentissage + Projection en une étape.

        Args:
            X (np.ndarray): Données d'entrée

        Returns:
            np.ndarray: Données projetées

        Example:
            >>> result = model.fit_transform(data)
        """

        self.fit(X)
        return self.transform(X)
    
    
    def plot_cov_matrix(self) -> None:
        """
        Affiche la matrice de covariance avec annotations.

        Example:
            >>> model.plot_cov_matrix()
        """

        plt.figure(figsize=(6,5))
        cov_img = plt.matshow(self.cov, cmap=plt.cm.Reds, fognum=1)
        plt.colorbar(cov_img, ticks=[-1, 0, 1])
        plt.xlabel("Variables")
        plt.ylabel("Variables")
        plt.title("Matrice de Covariance")
        for x in range(self.cov.shape[0]):
            for y in range(self.cov.shape[1]):
                plt.text(x, y, f"{self.cov[x, y]:.2f}" , size=8, color="black", ha="center", va="center")
        plt.show()

        
    def explained_variances(self) -> np.ndarray:
        """
        Retourne les valeurs propres triées.

        Returns:
            np.ndarray: Vecteur des valeurs propres

        Example:
            >>> print(model.explained_variances())
        """
        return self.eigen_values
    
    
    def explained_variances_ratio(self) -> np.ndarray:
        """
        Calcule le ratio de variance expliquée par composante.

        Returns:
            np.ndarray: Vecteur des ratios (somme à 1)

        Example:
            >>> print(model.explained_variances_ratio())
        """
        return self.eigen_values / np.sum(self.eigen_values)
    

    def plot_cumulative_explained_variance_ratio(self) -> None:
        """
        Affiche la courbe cumulative de variance expliquée.

        Example:
            >>> model.plot_cumulative_explained_variance_ratio()
        """
        
        plt.plot( 
            list(range(1, len(np.cumsum(self.explained_variances_ratio())) +1)),
            np.cumsum(self.explained_variances_ratio())
        )
        plt.xlabel("Nombre de composantes")
        plt.ylabel("% Explained Variance")
        plt.title("PCA Somme Cumulative Variance")
        plt.show()





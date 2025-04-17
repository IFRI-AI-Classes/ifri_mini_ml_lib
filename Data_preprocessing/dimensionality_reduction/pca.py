import numpy as np
import matplotlib.pyplot as plt

class ACP:

    """
    Implementation of Principal Component Analysis (PCA) for dimensionality reduction.

    Attributes:
        n_component (int): Number of principal components to retain
        Mean (np.ndarray): Mean of the features computed during training
        std (np.ndarray): Standard deviation of the features computed during training
        cov (np.ndarray): Computed covariance matrix
        eigen_values (np.ndarray): Sorted eigenvalues
        eigen_vectors (np.ndarray): Sorted eigenvectors
        components (np.ndarray): Selected principal components
    """

    def __init__(self, n_component: int):
       
        """
        Initializes the PCA model.

        Args:
            n_component (int): Number of principal components to keep

        Example:
            >>> model = PCA(n_component=2)
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
        Model training (calculation of statistics and components).

        Args:
            X (np.ndarray): 2D data matrix (shape: [n_samples, n_features])

        Returns:
            PCA: Trained instance

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
        Projection of data into the reduced space.

        Args:
            X (np.ndarray): Data to transform (shape: [n_samples, n_features])

        Returns:
            np.ndarray: Projected data (shape: [n_samples, n_component])

        Example:
            >>> transformed_data = model.transform(data)
            >>> print(transformed_data.shape)
            (3, 2)
        """

        X = (X - self.mean) / (self.std + 1e-10)
        return np.dot(X, self.components)
    
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        """
        One-step learning + projection.

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Projected data

        Example:
            >>> result = model.fit_transform(data)
        """

        self.fit(X)
        return self.transform(X)
    
    
    def plot_cov_matrix(self) -> None:
        
        """
        Displays the covariance matrix with annotations.

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
        Returns the sorted eigenvalues.

        Returns:
            np.ndarray: Vector of eigenvalues

        Example:
            >>> print(model.explained_variances())
        """
        return self.eigen_values
    
    
    def explained_variances_ratio(self) -> np.ndarray:     

        """
        Calculates the ratio of variance explained by component.

        Returns:
            np.ndarray: Vector of ratios (sum to 1)

        Example:
            >>> print(model.explained_variances_ratio())
        """
        return self.eigen_values / np.sum(self.eigen_values)
    

    def plot_cumulative_explained_variance_ratio(self) -> None:
       
        """
        Displays the cumulative explained variance ratio plot.

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


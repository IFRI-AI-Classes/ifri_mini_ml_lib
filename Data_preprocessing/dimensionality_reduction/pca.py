import numpy as np
import matplotlib.pyplot as plt

class ACP:
    def __init__(self, n_component):
        self.n_component = n_component
        self.mean = None
        self.std = None
        self.cov = None
        self.eigen_values = None
        self.eigen_vectors = None
        self.components = None

    def fit(self, X):

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

    def transform(self,X):
        X = (X - self.mean) / (self.std + 1e-10)
        return np.dot(X, self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def plot_cov_matrix(self):
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
        
    def explained_variances(self):
        return self.eigen_values
    
    def explained_variances_ratio(self):
        return self.eigen_values / np.sum(self.eigen_values)
    
    def plot_cumulative_explained_variance_ratio(self):
        plt.plot( 
            list(range(1, len(np.cumsum(self.explained_variances_ratio())) +1)),
            np.cumsum(self.explained_variances_ratio())
        )
        plt.xlabel("Nombre de composantes")
        plt.ylabel("% Explained Variance")
        plt.title("PCA Somme Cumulative Variance")
        plt.show()



####################     CODE ALGORITHMIQUE    ##############################


"""
function PCA(n_component) 
    mean ← NULL 
    std ← NULL 
    cov ← NULL 
    eigen_values ← NULL 
    eigen_vectors ← NULL 
    components ← NULL 

    function FIT(X) 
        // Centrage et normalisation des données
        mean ← CALCULER_MOYENNE(X) 
        std ← CALCULER_ECART_TYPE(X) 
        X ← (X - mean) / (std + 1e-10) 

        // Calcul de la matrice de covariance
        cov ← CALCULER_COVARIANCE(X) 

        // Décomposition en valeurs propres et vecteurs propres
        eigen_values, eigen_vectors ← DECOMPOSITION_PROPRE(cov) 

        // Tri des valeurs propres par ordre décroissant
        indices ← TRIER_DECROISSANT(eigen_values) 
        eigen_values ← eigen_values[indices] 
        eigen_vectors ← eigen_vectors[:, indices] 

        // Sélection des n premiers vecteurs propres
        components ← eigen_vectors[:, :n_component] 

        return self 
    end function

    function TRANSFORM(X) 
        X ← (X - mean) / (std + 1e-10) 
        return PRODUIT_SCALAIRE(X, components) 
    end function 

    function FIT_TRANSFORM(X) 
        FIT(X) 
        return TRANSFORM(X) 
    end function 

    function PLOT_COV_MATRIX() 
        // Affichage de la matrice de covariance
        IMPRIMER("Matrice de Covariance") 
        POUR i ← 0 à TAILLE(cov) - 1 FAIRE 
            POUR j ← 0 à TAILLE(cov) - 1 FAIRE 
                IMPRIMER("cov[", i, ",", j, "] = ", cov[i, j]) 
            FIN POUR 
        FIN POUR 
    end function 

    function EXPLAINED_VARIANCES() 
        return eigen_values 
    end function 

    function EXPLAINED_VARIANCES_RATIO() 
        total_variance ← SOMME(eigen_values) 
        variance_ratio ← eigen_values / total_variance 
        return variance_ratio 
    end function 

    function PLOT_CUMULATIVE_EXPLAINED_VARIANCE_RATIO() 
        variance_ratio ← EXPLAINED_VARIANCES_RATIO() 
        cumulative_variance ← CUMULATIVE_SOMME(variance_ratio) 

        IMPRIMER("PCA - Somme Cumulative de la Variance") 
        POUR k ← 1 à TAILLE(cumulative_variance) FAIRE 
            IMPRIMER("Composante ", k, ": ", cumulative_variance[k-1] * 100, "%") 
        FIN POUR 
    end function 

end function
"""



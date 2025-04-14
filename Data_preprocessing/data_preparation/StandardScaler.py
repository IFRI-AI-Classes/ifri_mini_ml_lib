#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Standardisation"""
import numpy as np
import pandas as pd

class StandardScaler:
    """
    Implémentation d'un standardiseur de données similaire à celui de scikit-learn.

    Cette classe permet de centrer et réduire les données (moyenne nulle, écart-type unitaire).
    Elle supporte aussi bien les tableaux numpy que les DataFrames pandas.

    Attributs :
    ----------
    mean_ : np.ndarray
        Moyenne de chaque colonne des données d'entraînement.

    std_ : np.ndarray
        Écart-type de chaque colonne des données d'entraînement.

    is_dataframe : bool
        Indique si les données d'origine étaient un DataFrame pandas.

    columns : Index or None
        Les noms des colonnes si les données d'origine étaient un DataFrame.
    """
    
    def __init__(self):
        """
        Initialise le StandardScaler avec des attributs vides.
        """
        self.mean_ = None
        self.std_ = None
        self.is_dataframe = False
        self.columns = None

    def _convert_to_array(self, X):
        """
        Convertit les données en tableau numpy si ce n'est pas déjà le cas.

        :param X: Données d'entrée (np.ndarray ou pd.DataFrame)
        :type X: np.ndarray or pd.DataFrame
        :return: Données converties en tableau numpy
        :rtype: np.ndarray
        """
        if isinstance(X, pd.DataFrame):
            self.is_dataframe = True
            self.columns = X.columns
            return X.values
        self.is_dataframe = False
        return np.array(X)

    def _convert_to_dataframe(self, X_scaled):
        """
        Convertit les données standardisées en DataFrame si les données initiales en étaient un.

        :param X_scaled: Données transformées
        :type X_scaled: np.ndarray
        :return: Données sous forme de DataFrame ou ndarray selon l'entrée d'origine
        :rtype: pd.DataFrame or np.ndarray
        """
        if self.is_dataframe:
            return pd.DataFrame(X_scaled, columns=self.columns)
        return X_scaled

    def fit(self, X):
        """
        Calcule la moyenne et l'écart-type des données.

        :param X: Données d'entrée (chaque ligne est un exemple)
        :type X: np.ndarray or pd.DataFrame
        """
        X = self._convert_to_array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)

    def transform(self, X):
        """
        Applique la transformation de standardisation aux données.

        :param X: Données à transformer
        :type X: np.ndarray or pd.DataFrame
        :return: Données standardisées
        :rtype: np.ndarray or pd.DataFrame
        :raises ValueError: Si 'fit' n'a pas été appelé auparavant
        """
        X = self._convert_to_array(X)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Le scaler n'a pas été ajusté. Appelez 'fit' avant 'transform'.")
        
        X_scaled = (X - self.mean_) / self.std_
        return self._convert_to_dataframe(X_scaled)

    def fit_transform(self, X):
        """
        Ajuste le scaler aux données puis applique la transformation.

        :param X: Données d'entrée
        :type X: np.ndarray or pd.DataFrame
        :return: Données standardisées
        :rtype: np.ndarray or pd.DataFrame
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """
        Annule la standardisation et récupère les valeurs d'origine.

        :param X_scaled: Données transformées
        :type X_scaled: np.ndarray or pd.DataFrame
        :return: Données ramenées à leur échelle d'origine
        :rtype: np.ndarray or pd.DataFrame
        :raises ValueError: Si 'fit' n'a pas été appelé auparavant
        """
        X_scaled = self._convert_to_array(X_scaled)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Le scaler n'a pas été ajusté. Appelez 'fit' avant 'inverse_transform'.")
        
        X_original = (X_scaled * self.std_) + self.mean_
        return self._convert_to_dataframe(X_original)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Normalisation"""
import numpy as np
import pandas as pd

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)): # initialisation de l'instance avec une plage de normalisation spécifiée avec feature_range
        self.min_ = None
        self.max_ = None
        self.range_min, self.range_max = feature_range

    def fit(self, X):
        """Calcule les min et max pour normaliser les données."""
        X = self._convert_to_array(X)
        self.min_ = np.min(X, axis=0)  # Min par colonne
        self.max_ = np.max(X, axis=0)  # Max par colonne

    def transform(self, X):
        """Applique la normalisation Min-Max sur les données."""
        X = self._convert_to_array(X)

        if self.min_ is None or self.max_ is None: #verifie que fit a été appelé
            raise ValueError("Le scaler n'a pas été ajusté. Appelez 'fit' avant 'transform'.")

        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        return self._convert_back(X_scaled, X)#convertit le résultat dans le meme format que les données d'origine

    def fit_transform(self, X):
        """Combine fit et transform en une seule étape."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """Rétablit les valeurs d'origine à partir des données normalisées."""
        X_scaled = self._convert_to_array(X_scaled)

        if self.min_ is None or self.max_ is None:
            raise ValueError("Le scaler n'a pas été ajusté. Appelez 'fit' avant 'inverse_transform'.")

        X_original = self.min_ + X_scaled * (self.max_ - self.min_)
        np.set_printoptions(suppress=True)  # Désactive la notation scientifique
        return self._convert_back(X_original, X_scaled)#convertit le résultat dans le meme format que les données d'origine

    def _convert_to_array(self, X):
        """Convertit l'entrée en tableau NumPy (gère listes, DataFrames et Series)."""
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.values  # Convertir en array NumPy
        return np.array(X)  # Si déjà NumPy ou liste, conversion directe

    def _convert_back(self, X, original):
        """Convertit l'array NumPy en DataFrame/Series si nécessaire."""
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(X, columns=original.columns)  # Rendre DataFrame
        if isinstance(original, pd.Series):
            return pd.Series(X)  # Rendre Series
        return pd.DataFrame(X)  



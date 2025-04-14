#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Normalisation"""
import numpy as np
import pandas as pd

class MinMaxScaler:
    """
    Implémentation du scaler Min-Max pour la normalisation des données.

    :param feature_range: Tuple représentant la plage désirée après normalisation (min, max)
    :type feature_range: tuple(float, float)
    """

    def __init__(self, feature_range=(0, 1)):
        """
        Initialise les attributs du scaler.

        :param feature_range: Plage de transformation souhaitée (par défaut entre 0 et 1)
        :type feature_range: tuple
        """
        self.min_ = None
        self.max_ = None
        self.range_min, self.range_max = feature_range

    def fit(self, X):
        """
        Calcule les valeurs minimales et maximales pour chaque feature.

        :param X: Données d'entraînement à normaliser
        :type X: array-like (list, np.ndarray, pd.DataFrame ou pd.Series)
        """
        X = self._convert_to_array(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)

    def transform(self, X):
        """
        Applique la normalisation Min-Max sur les données.

        :param X: Données à transformer
        :type X: array-like
        :return: Données normalisées dans l'intervalle spécifié
        :rtype: array-like (même type que l'entrée)
        :raises ValueError: Si fit() n'a pas été appelé avant
        """
        X = self._convert_to_array(X)

        if self.min_ is None or self.max_ is None:
            raise ValueError("Le scaler n'a pas été ajusté. Appelez 'fit' avant 'transform'.")

        X_scaled = (X - self.min_) / (self.max_ - self.min_)
        return self._convert_back(X_scaled, X)

    def fit_transform(self, X):
        """
        Applique successivement fit() puis transform().

        :param X: Données à normaliser
        :type X: array-like
        :return: Données normalisées
        :rtype: array-like
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """
        Reconvertit les données normalisées vers leurs valeurs d'origine.

        :param X_scaled: Données transformées à reconvertir
        :type X_scaled: array-like
        :return: Données originales reconstruites
        :rtype: array-like
        :raises ValueError: Si fit() n'a pas été appelé avant
        """
        X_scaled = self._convert_to_array(X_scaled)

        if self.min_ is None or self.max_ is None:
            raise ValueError("Le scaler n'a pas été ajusté. Appelez 'fit' avant 'inverse_transform'.")

        X_original = self.min_ + X_scaled * (self.max_ - self.min_)
        np.set_printoptions(suppress=True)
        return self._convert_back(X_original, X_scaled)

    def _convert_to_array(self, X):
        """
        Convertit l'entrée en tableau NumPy.

        :param X: Données sous forme liste, DataFrame ou Series
        :type X: list, pd.DataFrame, pd.Series, or np.ndarray
        :return: Données converties en array NumPy
        :rtype: np.ndarray
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return X.values
        return np.array(X)

    def _convert_back(self, X, original):
        """
        Convertit un array NumPy dans le même format que les données d'origine.

        :param X: Données transformées
        :type X: np.ndarray
        :param original: Données d'origine pour détecter le format
        :type original: array-like
        :return: Données dans le même format que l'entrée initiale
        :rtype: pd.DataFrame, pd.Series, or np.ndarray
        """
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(X, columns=original.columns)
        if isinstance(original, pd.Series):
            return pd.Series(X)
        return pd.DataFrame(X)


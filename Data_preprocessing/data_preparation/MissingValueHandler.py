#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Gestion des valeurs manquantes"""
import numpy as np
import pandas as pd

class MissingValueHandler:
    """
    Classe pour gérer les valeurs manquantes dans les datasets avec différentes méthodes.

    :param None: Cette classe ne prend pas de paramètres d'initialisation
    """

    def __init__(self):
        pass

    def _convert_to_dataframe(self, X):
        """
        Convertit l'input en pandas DataFrame si ce n'est pas déjà le cas.

        :param X: Données d'entrée (DataFrame, array numpy ou autre)
        :type X: pd.DataFrame or np.ndarray or array-like
        :return: Données converties en DataFrame pandas
        :rtype: pd.DataFrame
        """
        if isinstance(X, pd.DataFrame):
            return X  # Retourne directement le DataFrame
        return pd.DataFrame(X)  # Si ce n'est pas un DataFrame, on le transforme en DataFrame

    def remove_missing(self, X, threshold=0.5, axis=0):
        """
        Supprime les lignes ou colonnes avec trop de valeurs manquantes.

        :param X: Données d'entrée
        :type X: pd.DataFrame or np.ndarray
        :param threshold: Seuil de valeurs manquantes (0-1) au-delà duquel on supprime
        :type threshold: float
        :param axis: Axe sur lequel appliquer la suppression (0=lignes, 1=colonnes)
        :type axis: int
        :return: Données nettoyées
        :rtype: pd.DataFrame
        """
        df = self._convert_to_dataframe(X)
        df_cleaned = df.dropna(thresh=int(threshold * df.shape[axis]), axis=axis)
        return df_cleaned

    def impute_statistical(self, X, strategy="mean"):
        """
        Remplace les NaN par une valeur statistique (mean, median, mode).

        :param X: Données d'entrée
        :type X: pd.DataFrame or np.ndarray
        :param strategy: Stratégie d'imputation ('mean', 'median' ou 'mode')
        :type strategy: str
        :return: Données imputées
        :rtype: pd.DataFrame
        """
        df = self._convert_to_dataframe(X)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    def impute_default(self, X, value=0):
        """
        Remplace les valeurs NaN par une valeur fixe.

        :param X: Données d'entrée
        :type X: pd.DataFrame or np.ndarray
        :param value: Valeur de remplacement
        :type value: int or float
        :return: Données imputées
        :rtype: pd.DataFrame
        """
        df = self._convert_to_dataframe(X)
        return df.fillna(value)

    def impute_knn(self, X, k=3):
        """
        Imputation des valeurs manquantes à partir des k voisins les plus proches.

        :param X: Données d'entrée
        :type X: pd.DataFrame or np.ndarray
        :param k: Nombre de voisins à considérer
        :type k: int
        :return: Données imputées
        :rtype: pd.DataFrame
        """
        from scipy.spatial.distance import cdist

        df = self._convert_to_dataframe(X)

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                non_missing = df[df[col].notnull()]
                missing = df[df[col].isnull()]

                if len(non_missing) == 0:
                    continue
                
                distances = cdist(missing.drop(col, axis=1).fillna(0),
                                  non_missing.drop(col, axis=1).fillna(0), metric='euclidean')

                nearest_indices = np.argsort(distances, axis=1)[:, :k]
                knn_values = non_missing.iloc[nearest_indices.flatten()][col].values.reshape(-1, k)
                imputed_values = np.nanmean(knn_values, axis=1)

                df.loc[df[col].isnull(), col] = imputed_values
        return df

    def impute_regression(self, X, target_col):
        """
        Imputation par régression : prédit les valeurs manquantes d'une colonne avec les autres colonnes.

        :param X: Données d'entrée
        :type X: pd.DataFrame or np.ndarray
        :param target_col: Nom de la colonne cible à imputer
        :type target_col: str
        :return: Données imputées
        :rtype: pd.DataFrame
        :raises ValueError: Si la colonne cible n'existe pas dans X
        """
        from sklearn.linear_model import LinearRegression  
        df = self._convert_to_dataframe(X)

        if target_col not in df.columns:
            raise ValueError(f"La colonne '{target_col}' n'existe pas dans X.")

        known = df[df[target_col].notnull()]
        unknown = df[df[target_col].isnull()]

        if len(unknown) == 0:
            return df 

        model = LinearRegression()
        model.fit(known.drop(target_col, axis=1), known[target_col])
        predicted_values = model.predict(unknown.drop(target_col, axis=1))

        df.loc[df[target_col].isnull(), target_col] = predicted_values
        return df


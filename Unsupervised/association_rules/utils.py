import pandas as pd
import numpy as np
from typing import Union, List, Set, Optional

class DataAdapter:
    """
    Classe utilitaire pour convertir différents formats de données en transactions
    pour l'algorithme Apriori.
    """
    
    @staticmethod
    def convert_to_transactions(data: Union[pd.DataFrame, np.ndarray, List], 
                               binary_mode: bool = False, 
                               columns: Optional[List[str]] = None,
                               separator: str = "_") -> List[Set[str]]:
        """
        Convertit différents types de données en transactions pour Apriori.
        
        Args:
            data: Source de données (DataFrame, NumPy array ou liste)
            binary_mode: Si True, considère les valeurs 1/True comme présence d'item
            columns: Liste des colonnes à considérer (uniquement pour DataFrame)
            separator: Séparateur utilisé pour joindre le nom des attributs et leurs valeurs
        
        Returns:
            Liste de transactions où chaque transaction est un ensemble d'items
            
        Raises:
            TypeError: Si le type de données n'est pas supporté
            ValueError: Si les données sont vides ou mal formées
        """
        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            raise ValueError("Les données ne peuvent pas être vides")
            
        # Conversion depuis pandas DataFrame
        if isinstance(data, pd.DataFrame):
            return DataAdapter._convert_from_dataframe(data, binary_mode, columns, separator)
            
        # Conversion depuis numpy array
        elif isinstance(data, np.ndarray):
            return DataAdapter._convert_from_numpy(data, binary_mode, separator)
            
        # Conversion depuis liste
        elif isinstance(data, list):
            return DataAdapter._convert_from_list(data)
            
        else:
            raise TypeError("Type de données non supporté. Formats acceptés: pandas DataFrame, numpy array, liste de transactions")
    
    @staticmethod
    def _convert_from_dataframe(df: pd.DataFrame, 
                               binary_mode: bool, 
                               columns: Optional[List[str]],
                               separator: str) -> List[Set[str]]:
        """
        Convertit un DataFrame pandas en liste de transactions.
        
        Args:
            df: DataFrame source
            binary_mode: Indique si les colonnes sont binaires (1/True = présent)
            columns: Liste des colonnes à utiliser (None = toutes les colonnes)
            separator: Séparateur entre nom d'attribut et valeur
            
        Returns:
            Liste de transactions
        """
        # Sélectionner les colonnes si spécifiées
        if columns:
            df = df[columns]
        
        # Vérifier s'il y a des valeurs manquantes
        has_missing = df.isna().any().any()
        
        if binary_mode:
            # Pour données binaires: chaque colonne où valeur = 1/True devient un item
            return [
                {f"{col}" for col, val in row.items() 
                 if val == 1 or val is True} 
                for _, row in df.iterrows()
            ]
        else:
            # Pour données catégorielles: format "colonne_valeur" pour chaque item
            return [
                {f"{col}{separator}{str(val).strip()}" for col, val in row.items() 
                 if pd.notna(val) and str(val).strip() != ""} 
                for _, row in df.iterrows()
            ]
    
    @staticmethod
    def _convert_from_numpy(arr: np.ndarray, 
                           binary_mode: bool,
                           separator: str) -> List[Set[str]]:
        """
        Convertit un array NumPy en liste de transactions.
        
        Args:
            arr: Array source
            binary_mode: Indique si l'array est binaire (1/True = présent)
            separator: Séparateur entre nom d'attribut et valeur
            
        Returns:
            Liste de transactions
        """
        if arr.ndim < 2:
            # Convertir un array 1D en array 2D
            arr = arr.reshape(1, -1)
            
        if binary_mode:
            # Pour data binaire: numéro de colonne devient l'item si valeur = 1/True
            return [
                {f"feature_{j}" for j in range(arr.shape[1]) 
                 if row[j] == 1 or row[j] is True} 
                for row in arr
            ]
        else:
            # Pour data non-binaire: format "feature_i_valeur" pour chaque item
            return [
                {f"feature_{j}{separator}{val}" for j, val in enumerate(row) 
                 if val is not None and not (isinstance(val, float) and np.isnan(val))
                 and str(val).strip() != ""} 
                for row in arr
            ]
    
    @staticmethod
    def _convert_from_list(data: List) -> List[Set[str]]:
        """
        Convertit une liste en liste de transactions.
        Suppose que l'entrée est déjà une liste de listes/ensembles d'items.
        
        Args:
            data: Liste source (liste de listes/ensembles d'items)
            
        Returns:
            Liste de transactions
            
        Raises:
            ValueError: Si la structure de la liste n'est pas adaptée
        """
        # Vérifier que l'input a la bonne structure
        if not data:
            return []
            
        # Conversion en liste d'ensembles
        transactions = []
        for transaction in data:
            # Si c'est déjà un ensemble, utiliser tel quel
            if isinstance(transaction, set):
                transactions.append(transaction)
            # Si c'est une liste ou un tuple, convertir en ensemble
            elif isinstance(transaction, (list, tuple)):
                # Filtrer les valeurs vides ou None
                clean_transaction = {str(item).strip() for item in transaction 
                                    if item is not None and str(item).strip() != ""}
                if clean_transaction:  # Ne pas ajouter d'ensembles vides
                    transactions.append(clean_transaction)
            else:
                # Si élément unique, créer un ensemble avec cet élément
                if transaction is not None and str(transaction).strip() != "":
                    transactions.append({str(transaction).strip()})
                
        return transactions
    
    @staticmethod
    def load_csv_to_transactions(file_path: str, header: Optional[int] = None, 
                                separator: str = ',') -> List[Set[str]]:
        """
        Charge un fichier CSV et le convertit directement en transactions.
        
        Args:
            file_path: Chemin vers le fichier CSV
            header: Numéro de ligne d'en-tête (None = pas d'en-tête)
            separator: Séparateur de colonnes dans le fichier CSV
            
        Returns:
            Liste de transactions
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le fichier est vide ou mal formé
        """
        try:
            # Charger le CSV comme DataFrame
            df = pd.read_csv(file_path, header=header, sep=separator)
            
            # Convertir chaque ligne en ensemble d'items (en ignorant les valeurs manquantes)
            transactions = []
            for _, row in df.iterrows():
                # Filtrer les valeurs non-null/non-empty
                transaction = {str(item).strip() for item in row.dropna() 
                              if str(item).strip() != ""}
                if transaction:  # Ne pas ajouter de transactions vides
                    transactions.append(transaction)
                    
            return transactions
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Le fichier {file_path} n'a pas été trouvé")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier CSV: {str(e)}")
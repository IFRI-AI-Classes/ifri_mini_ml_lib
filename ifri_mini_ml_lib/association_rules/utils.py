import pandas as pd
import numpy as np
from typing import Union, List, Set, Optional

class DataAdapter:
    """
    Utility class for converting different data formats into transactions
    for association rule mining algorithms.
    
    Examples:
    
    >>> # Converting from a pandas DataFrame
    >>> import pandas as pd
    >>> from ifri_mini_ml_lib.association_rules.utils import DataAdapter
    >>> # Create a sample DataFrame
    >>> df = pd.DataFrame({
    ...     'item1': [1, 0, 1, 1, 0],
    ...     'item2': [1, 1, 0, 1, 1],
    ...     'item3': [0, 1, 1, 1, 0]
    ... })
    >>> # Convert to transactions (binary mode)
    >>> transactions = DataAdapter.convert_to_transactions(df, binary_mode=True)
    >>> for t in transactions:
    ...     print(sorted(t))
    ['item1', 'item2']
    ['item2', 'item3']
    ['item1', 'item3']
    ['item1', 'item2', 'item3']
    ['item2']
    
    >>> # Converting non-binary data
    >>> df_cat = pd.DataFrame({
    ...     'color': ['red', 'blue', 'green', 'red', 'blue'],
    ...     'size': ['large', 'medium', 'small', 'medium', 'large']
    ... })
    >>> # Convert to transactions (categorical data)
    >>> transactions = DataAdapter.convert_to_transactions(df_cat, binary_mode=False)
    >>> for t in transactions[:2]:
    ...     print(sorted(t))
    ['color_red', 'size_large']
    ['color_blue', 'size_medium']
    
    >>> # Converting from a list of sets
    >>> list_data = [
    ...     {'bread', 'milk'},
    ...     {'bread', 'diaper', 'beer'},
    ...     {'milk', 'diaper', 'beer'},
    ...     {'bread', 'milk', 'diaper'},
    ...     {'bread', 'milk', 'beer'}
    ... ]
    >>> transactions = DataAdapter._convert_from_list(list_data)
    >>> print(len(transactions), "transactions loaded")
    5 transactions loaded
    """
    
    @staticmethod
    def convert_to_transactions(data: Union[pd.DataFrame, np.ndarray, List], 
                               binary_mode: bool = False, 
                               columns: Optional[List[str]] = None,
                               separator: str = "_") -> List[Set[str]]:
        """
        Converts different data types into transactions for association rule mining.
        
        Args:
            data: Data source (DataFrame, NumPy array or list)
            binary_mode: If True, considers 1/True values as item presence
            columns: List of columns to consider (only for DataFrame)
            separator: Separator used to join attribute names and their values
        
        Returns:
            List of transactions where each transaction is a set of items
            
        Raises:
            TypeError: If the data type is not supported
            ValueError: If data is empty or malformed
        """
        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            raise ValueError("Data cannot be empty")
            
        # Convert from pandas DataFrame
        if isinstance(data, pd.DataFrame):
            return DataAdapter._convert_from_dataframe(data, binary_mode, columns, separator)
            
        # Convert from numpy array
        elif isinstance(data, np.ndarray):
            return DataAdapter._convert_from_numpy(data, binary_mode, separator)
            
        # Convert from list
        elif isinstance(data, list):
            return DataAdapter._convert_from_list(data)
            
        else:
            raise TypeError("Unsupported data type. Accepted formats: pandas DataFrame, numpy array, list of transactions")
    
    @staticmethod
    def _convert_from_dataframe(df: pd.DataFrame, 
                               binary_mode: bool, 
                               columns: Optional[List[str]],
                               separator: str) -> List[Set[str]]:
        """
        Converts a pandas DataFrame into a list of transactions.
        
        Args:
            df: Source DataFrame
            binary_mode: Indicates if columns are binary (1/True = present)
            columns: List of columns to use (None = all columns)
            separator: Separator between attribute name and value
            
        Returns:
            List of transactions
        """
        # Select columns if specified
        if columns:
            df = df[columns]
        
        # Check for missing values
        has_missing = df.isna().any().any()
        
        if binary_mode:
            # For binary data: each column where value = 1/True becomes an item
            return [
                {f"{col}" for col, val in row.items() 
                 if val == 1 or val is True} 
                for _, row in df.iterrows()
            ]
        else:
            # For categorical data: format "column_value" for each item
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
        Converts a NumPy array into a list of transactions.
        
        Args:
            arr: Source array
            binary_mode: Indicates if the array is binary (1/True = present)
            separator: Separator between attribute name and value
            
        Returns:
            List of transactions
        """
        if arr.ndim < 2:
            # Convert a 1D array into a 2D array
            arr = arr.reshape(1, -1)
            
        if binary_mode:
            # For binary data: column number becomes the item if value = 1/True
            return [
                {f"feature_{j}" for j in range(arr.shape[1]) 
                 if row[j] == 1 or row[j] is True} 
                for row in arr
            ]
        else:
            # For non-binary data: format "feature_i_value" for each item
            return [
                {f"feature_{j}{separator}{val}" for j, val in enumerate(row) 
                 if val is not None and not (isinstance(val, float) and np.isnan(val))
                 and str(val).strip() != ""} 
                for row in arr
            ]
    
    @staticmethod
    def _convert_from_list(data: List) -> List[Set[str]]:
        """
        Converts a list into a list of transactions.
        Assumes that the input is already a list of lists/sets of items.
        
        Args:
            data: Source list (list of lists/sets of items)
            
        Returns:
            List of transactions
            
        Raises:
            ValueError: If the list structure is not suitable
        """
        # Check that the input has the right structure
        if not data:
            return []
            
        # Convert to list of sets
        transactions = []
        for transaction in data:
            # If it's already a set, use as is
            if isinstance(transaction, set):
                transactions.append(transaction)
            # If it's a list or tuple, convert to set
            elif isinstance(transaction, (list, tuple)):
                # Filter empty values or None
                clean_transaction = {str(item).strip() for item in transaction 
                                    if item is not None and str(item).strip() != ""}
                if clean_transaction:  # Don't add empty sets
                    transactions.append(clean_transaction)
            else:
                # If single element, create a set with this element
                if transaction is not None and str(transaction).strip() != "":
                    transactions.append({str(transaction).strip()})
                
        return transactions
    
    @staticmethod
    def load_csv_to_transactions(file_path: str, header: Optional[int] = None, 
                                separator: str = ',') -> List[Set[str]]:
        """
        Loads a CSV file and converts it directly into transactions.
        
        Args:
            file_path: Path to the CSV file
            header: Header row number (None = no header)
            separator: Column separator in the CSV file
            
        Returns:
            List of transactions
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or malformed
        """
        try:
            # Load CSV as DataFrame
            df = pd.read_csv(file_path, header=header, sep=separator)
            
            # Convert each row to a set of items (ignoring missing values)
            transactions = []
            for _, row in df.iterrows():
                # Filter non-null/non-empty values
                transaction = {str(item).strip() for item in row.dropna() 
                              if str(item).strip() != ""}
                if transaction:  # Don't add empty transactions
                    transactions.append(transaction)
                    
            return transactions
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found")
        except Exception as e:
            raise ValueError(f"Error while loading CSV file: {str(e)}")
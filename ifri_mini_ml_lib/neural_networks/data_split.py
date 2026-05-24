import numpy as np
from typing import Tuple
from ..preprocessing.preparation.splitting import DataSplitter
import pandas as pd


# Data splitting function for training and validation sets
def split_train_validation( X: np.ndarray, y: np.ndarray, validation_fraction: float,  seed = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    y : np.ndarray
        Target values or classes
    seed : int or None
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_val, y_train, y_val : The split datasets
    """

    # Convert to pandas DataFrame for compatibility with DataSplitter
    X_df = pd.DataFrame(X)
    y_df = pd.Series(y)

    splitter = DataSplitter(seed=seed)
    X_train, X_val, y_train, y_val = splitter.train_test_split(X_df, y_df, validation_fraction)
    
    # Convert back to numpy arrays
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    return X_train, X_val, y_train, y_val
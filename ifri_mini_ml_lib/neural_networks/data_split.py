import numpy as np
from typing import Tuple
from ..preprocessing.preparation.splitting import DataSplitter


# Data splitting function for training and validation sets
def split_train_validation( X: np.ndarray, y: np.ndarray, validation_fraction: int,  seed = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    splitter = DataSplitter(seed=seed)
    X_train, X_val, y_train, y_val = splitter.train_test_split(X, y, validation_fraction)
    
    return X_train, X_val, y_train, y_val
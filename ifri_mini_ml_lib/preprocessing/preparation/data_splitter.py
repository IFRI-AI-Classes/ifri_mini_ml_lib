import numpy as np

class DataSplitter:
    def __init__(self, seed=None):
        """
        Initialize the DataSplitter with an optional random seed.
        
        Args:
            seed (int, optional): Random seed for reproducible splits. Defaults to None.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def train_test_split(self, X, y=None, test_size=0.2, val_set=False, val_size=0.2):
        """
        Split data randomly into train and test sets, with optional validation set.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series, optional): Target data. Defaults to None.
            test_size (float, optional): Proportion of dataset to include in test split. Defaults to 0.2.
            val_set (bool, optional): Whether to create validation set. Defaults to False.
            val_size (float, optional): Proportion of train set to include in validation split. Defaults to 0.2.
            
        Returns:
            If y is None:
                X_train, X_test or X_train, X_val, X_test if val_set=True
            If y is provided:
                X_train, X_test, y_train, y_test or 
                X_train, X_val, X_test, y_train, y_val, y_test if val_set=True
                
        Example:
            >>> splitter = DataSplitter(seed=42)
            >>> X_train, X_test = splitter.train_test_split(X)
            >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_test_split(
            ...     X, y, test_size=0.3, val_set=True, val_size=0.1)
        """
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        test_end = int(test_size * len(X))
        test_idx = indices[:test_end]
        train_idx = indices[test_end:]
        
        if not val_set:
            if y is None:
                return X.iloc[train_idx], X.iloc[test_idx]
            else:
                return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
        else:
            val_split = int(val_size * len(train_idx))
            val_idx = train_idx[:val_split]
            new_train_idx = train_idx[val_split:]
            
            if y is None:
                return X.iloc[new_train_idx], X.iloc[val_idx], X.iloc[test_idx]
            else:
                return (X.iloc[new_train_idx], X.iloc[val_idx], X.iloc[test_idx],
                        y.iloc[new_train_idx], y.iloc[val_idx], y.iloc[test_idx])

    def stratified_train_test_split(self, X, y, test_size=0.2, val_set=False, val_size=0.2):
        """
        Split data into train and test sets preserving class distribution, with optional validation set.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data (required for stratified split)
            test_size (float, optional): Proportion of dataset to include in test split. Defaults to 0.2.
            val_set (bool, optional): Whether to create validation set. Defaults to False.
            val_size (float, optional): Proportion of train set to include in validation split. Defaults to 0.2.
            
        Returns:
            If val_set=False:
                X_train, X_test, y_train, y_test
            If val_set=True:
                X_train, X_val, X_test, y_train, y_val, y_test
                
        Example:
            >>> splitter = DataSplitter(seed=42)
            >>> X_train, X_test, y_train, y_test = splitter.stratified_train_test_split(X, y)
            >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.stratified_train_test_split(
            ...     X, y, val_set=True, val_size=0.1)
        """
        assert y is not None, "Stratified split requires target vector y"
        
        classes = np.unique(y)
        train_idx, val_idx, test_idx = [], [], []
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            n = len(cls_indices)
            
            test_end = int(test_size * n)
            cls_test = cls_indices[:test_end]
            cls_train = cls_indices[test_end:]
            
            if val_set:
                val_split = int(val_size * len(cls_train))
                cls_val = cls_train[:val_split]
                cls_train = cls_train[val_split:]
                val_idx.extend(cls_val)
            
            test_idx.extend(cls_test)
            train_idx.extend(cls_train)
        
        if not val_set:
            return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
        else:
            return (X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx],
                    y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx])

    def temporal_train_test_split(self, X, y=None, test_size=0.2, val_set=False, val_size=0.2):
        """
        Split time series data into train and test sets (and optionally validation),
        preserving temporal order.
        
        Args:
            X (pd.DataFrame): Feature data (time-ordered)
            y (pd.Series, optional): Target data (time-ordered)
            test_size (float, optional): Proportion of dataset to include in test split. Defaults to 0.2.
            val_set (bool, optional): Whether to create validation set. Defaults to False.
            val_size (float, optional): Proportion of train set to include in validation split. Defaults to 0.2.
            
        Returns:
            If y is None:
                X_train, X_test or X_train, X_val, X_test if val_set=True
            If y is provided:
                X_train, X_test, y_train, y_test or 
                X_train, X_val, X_test, y_train, y_val, y_test if val_set=True
                
        Example:
            >>> splitter = DataSplitter()
            >>> X_train, X_test = splitter.temporal_train_test_split(X)
            >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.temporal_train_test_split(
            ...     X, y, val_set=True)
        """
        n_samples = len(X)
        test_end = int(test_size * n_samples)
        
        X_test = X.iloc[-test_end:]  # Take most recent samples for test
        X_train = X.iloc[:-test_end]
        
        if y is None:
            if not val_set:
                return X_train, X_test
            else:
                val_end = int(val_size * len(X_train))
                X_val = X_train.iloc[-val_end:]
                X_train = X_train.iloc[:-val_end]
                return X_train, X_val, X_test
        else:
            y_test = y.iloc[-test_end:]
            y_train = y.iloc[:-test_end]
            
            if not val_set:
                return X_train, X_test, y_train, y_test
            else:
                val_end = int(val_size * len(X_train))
                X_val = X_train.iloc[-val_end:]
                y_val = y_train.iloc[-val_end:]
                X_train = X_train.iloc[:-val_end]
                y_train = y_train.iloc[:-val_end]
                return X_train, X_val, X_test, y_train, y_val, y_test

    def k_fold_split(self, X, y=None, k=5):
        """
        Generate k folds for cross-validation.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series, optional): Target data. Defaults to None.
            k (int, optional): Number of folds. Defaults to 5.
            
        Returns:
            list: List of tuples containing train-test splits. Each tuple contains:
                If y is None: (X_train, X_test)
                If y is provided: (X_train, X_test, y_train, y_test)
                
        Example:
            >>> splitter = DataSplitter(seed=42)
            >>> folds = splitter.k_fold_split(X, y, k=5)
            >>> for X_train, X_test, y_train, y_test in folds:
            ...     # Train and evaluate model
        """
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        fold_size = len(X) // k
        folds = []
        
        for i in range(k):
            test_idx = indices[i * fold_size:(i + 1) * fold_size]
            train_idx = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
            if y is None:
                folds.append((X.iloc[train_idx], X.iloc[test_idx]))
            else:
                folds.append((X.iloc[train_idx], X.iloc[test_idx],
                             y.iloc[train_idx], y.iloc[test_idx]))
        return folds
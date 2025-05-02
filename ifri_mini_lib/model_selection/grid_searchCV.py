import numpy as np
from .utils import clone
from itertools import product
from .cross_validation import k_fold_cross_validation

class GridSearchCV:
    """
    Grid search implementation with cross-validation for hyperparameter optimization.
    
    This class allows to test different combinations of hyperparameters on a machine learning model
    using cross-validation to avoid overfitting and select the best hyperparameter combination.
    
    Args:
        model: ML model to optimize (must implement fit and predict methods)
        param_grid: Dictionary of hyperparameters to test, where keys are parameter 
                    names and values are lists of parameter values to try
        scoring: Evaluation function (e.g., accuracy_score, mean_squared_error, f1)
        k: Number of folds for cross-validation (default: 5)
        stratified: Whether to use stratified k-fold validation (default: False)
        
    Returns:
        None
        
    Example:
        >>> from ifri_mini_lib.supervised.classification import SVC
        >>> from ifri_mini_lib.metrics import accuracy_score
        >>> model = SVC()
        >>> param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        >>> grid_search = GridSearchCV(model, param_grid, accuracy_score, k=5)
    
    """

    def __init__(self, model, param_grid, scoring, k=5, stratified=False):
        self.model = model
        self.param_grid = param_grid
        self.k = k
        self.scoring = scoring 
        self.stratified = stratified
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

    
    def fit(self, X, y):
        """
        Description:
            Run grid search with cross-validation on the given data.
            Tests all possible combinations of hyperparameters and finds the best set.
            
        Args:
            X: Input features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
            
        Returns:
            self: The instance itself, allowing for method chaining
            
        Example:
            >>> grid_search.fit(X_train, y_train)
            >>> best_params = grid_search.best_params_
            >>> best_score = grid_search.best_score_
            >>> best_model = grid_search.best_estimator_
        """
        # Extract hyperparameter names and values
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # Generate all possible combinations of hyperparameters
        for param_combination in product(*param_values):
            # Create parameter dictionary for current combination
            params = dict(zip(param_names, param_combination))
            # Clone the model and set parameters
            model = clone(self.model).set_params(**params)
            
            # Perform cross-validation (stratified or standard)
            if self.stratified:
                mean_score, _ = k_fold_cross_validation(self.model, X, y, metric=self.scoring, stratified=True, k=self.k)
            else: 
                mean_score, _ = k_fold_cross_validation(self.model, X, y, metric=self.scoring, stratified=False, k=self.k)

            # Update best parameters if current combination is better
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_estimator_ = model.fit(X, y) 

        #print("Best hyperparameters:", self.best_params_)
        #print("Best score:", self.best_score_)
        
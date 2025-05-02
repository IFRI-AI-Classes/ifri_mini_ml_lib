import numpy as np
from itertools import product
from .utils import clone
from .cross_validation import k_fold_cross_validation

class RandomSearchCV:
    """
    Description:
        A manual implementation of Random Search combined with Cross-Validation.
        This class randomly samples a subset of all hyperparameter combinations
        and evaluates each using k-fold cross-validation. It selects the model
        with the best average score based on the provided scoring function.

    Args:
        model (object): The machine learning model to optimize (must implement `fit`, `predict`, `set_params`).
        
        param_grid (dict): Dictionary of hyperparameters to explore. 
                           Format: {'param1': [v1, v2], 'param2': [v1, v2, v3]}.
        
        scoring (callable): Scoring function to evaluate model performance. 
                            Example: ifri_mini_lib.metrics.accuracy_score or custom function.
        
        n_iter (int, optional): Number of random parameter combinations to try. Default is 10.
        
        k (int, optional): Number of folds for k-fold cross-validation. Default is 5.
        
        stratified (bool, optional): Whether to use stratified k-folds (useful for classification). Default is False.
        random_state (int or None, optional): Random seed for reproducibility.

    Attributes:
        best_params_ (dict): The best set of hyperparameters found during search.
        best_score_ (float): The best cross-validated score obtained.
        best_estimator_ (object): A clone of the model with the best hyperparameters.
    
    Example:
        >>> from ifri_mini_lib.supervised.classification import DecisionTreeClassifier
        >>> from ifri_mini_lib.metrics import accuracy_score

        >>> param_grid = {
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 4, 6]
            }

        >>> search = RandomSearchCV_Scratch(
            model=DecisionTreeClassifier(),
            param_grid=param_grid,
            scoring=accuracy_score,
            n_iter=5,
            k=3,
            stratified=True,
            random_state=42
            )

        >>> search.fit(X_train, y_train)
        >>> best_model = search.best_estimator_
        >>> predictions = best_model.predict(X_test)
    """
    def __init__(self, model, param_grid, scoring, n_iter=10, k=5, stratified = False, random_state=None):
        self.model = model
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.k = k
        self.scoring = scoring 
        self.stratified = stratified
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

    
    def fit(self, X, y):
        """
        Description:
            Runs the random search with cross-validation over the given dataset.
            Randomly selects hyperparameter combinations and evaluates them.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Target vector.

        Returns:
            None

        Example:
            >>> search.fit(X_train, y_train)
        """

        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_combinations = list(product(*param_values))
        
        rng = np.random.default_rng(self.random_state)
        
        #Tire au hasard (et sans doublons) des indices parmi toutes les combinaisons d’hyperparamètres disponibles, dans la limite de n_iter.
        sampled_combinations = rng.choice(len(all_combinations), size=min(self.n_iter, len(all_combinations)), replace=False)

        for idx in sampled_combinations:
            params = dict(zip(param_names, all_combinations[idx]))
           
            model = clone(self.model)
            model.set_params(**params)

            if self.stratified:
                mean_score, _ = k_fold_cross_validation(model, X, y, metric = self.scoring, stratified=True, k=self.k)
            else: 
                mean_score, _ = k_fold_cross_validation(model, X, y, metric = self.scoring, stratified=False, k=self.k)
            
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_estimator_ = model

    
    

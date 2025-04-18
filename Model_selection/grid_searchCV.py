import numpy as np
from utils import clone
from itertools import product
from cross_validation import k_fold_cross_validation

class GridSearchCV_Scratch:
    def __init__(self, model, param_grid, scoring,  k=5, stratified = False):
        """
        Initialise la recherche par grille.
        - model : modèle ML
        - param_grid : dictionnaire des hyperparamètres à tester
        - k : nombre de folds pour la Cross Validation
        - scoring : fonction d’évaluation (ex: accuracy_score, mean_squared_error, f1)
        - stratified: La version stratifiée ou non de k-fold validation
        """

        self.model = model
        self.param_grid = param_grid
        self.k = k
        self.scoring = scoring 
        self.stratified = stratified
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

    
    def fit(self, X, y):
        """Lance Grid Search avec Cross Validation"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            model = clone(self.model).set_params(**params)
            
            if self.stratified:
                mean_score, _ = k_fold_cross_validation(self.model, X, y, metric = self.scoring, stratified=True, k=self.k)
            else: 
                mean_score, _ = k_fold_cross_validation(self.model, X, y, metric = self.scoring, stratified=False, k=self.k)

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_estimator_ = model.fit(X, y) 

        #print("Meilleurs hyperparamètres :", self.best_params_)
        #print("Meilleur score :", self.best_score_)
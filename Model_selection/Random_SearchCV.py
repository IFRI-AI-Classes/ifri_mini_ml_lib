import numpy as np
from itertools import product
from utils import clone
from CrossValidation import k_fold_cross_validation

class RandomSearchCV_Scratch:
    def __init__(self, model, param_grid, scoring, n_iter=10, k=5, stratified = False, random_state=None):
        """
        Initialise la recherche aléatoire.
        - model : modèle ML
        - param_grid : dictionnaire des hyperparamètres à explorer
        - n_iter : nombre de combinaisons à tester
        - k : nombre de folds pour la Cross Validation
        - scoring : fonction d évaluation
        - stratified : Cross Validation stratifiée ou non
        - random_state : pour la reproductibilité
        """
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

    
    

import numpy as np 
from collections import defaultdict 
def k_fold_cross_validation (model, X, y, metric, stratified, k = 5):
        """
        Implémentation de la validation croisée k-Fold

        Parameters:
        - model : ML model with .set_params(), .fit() and .predict() methods
        - X : Features (list or numpy array)
        - y : Target (list or numpy array)
        - k : Number of folds (default, 5)
        - metric : (accuracy, mse, f1-score etc..)

        returns :
        - k-fold Mean score, 
        - écart type

        """
        
        X, y = np.array(X), np.array(y) 
        n_samples = len(X)

        #Gestion des erreurs sur le nbre de folds
        if k > n_samples or k < 2:
            raise ValueError("Le nombre de folds doit être entre 2 et le nombre total d'échantillons")
    
        if stratified:
            class_indices = defaultdict(list)
            for idx, label in enumerate(y):
                class_indices[label].append(idx)

            for idx in class_indices.values():
                np.random.seed(42)
                np.random.shuffle(idx)
            
            folds = [[] for _ in range (k)]
            for label, label_indices in class_indices.items():
                for i, idx in enumerate(label_indices):
                    folds[i % k].append(idx)
            
            folds = [np.array(fold) for fold in folds]
           
            indices = np.concatenate([fold for fold in folds])
            

        
        else: 
            #Melanger les indices pour eviter les biais liés à l'ordre des données et améliorer la généralisation du modele
            indices = np.arange(n_samples) # array ([0, 1, 2, 3, ..., n_samples-1]
            np.random.seed(42)
            np.random.shuffle(indices)

            fold_size = n_samples // k

            #formation des folds par indices
            #utilisation d'une list comprehension: ['expression' for 'element' in 'iterable' if 'condition']
            folds = [indices[i * fold_size : (i+1) * fold_size] for i in range(k)]
            if (n_samples % k) != 0 :
                folds[-1] = np.concatenate([folds[-1], indices[k * fold_size:]])

        #Validation Croisée
        scores = []

        for test_indices in folds:
            #separons à present chaque fold en train/test
            #train_indices = [idx for idx in indices if idx not in test_indices] #coûteux
            #train_indices = list(set(indices) - set(test_indices)) #bien mais perd l'ordre et peut-être coûteux
            
            train_indices = np.setdiff1d(indices, test_indices, assume_unique=True) #operation vectorielle optimisée

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            if not (hasattr(model, "fit") and hasattr(model, "predict")):
                raise ValueError("Le modèle doit avoir les méthodes .fit() et .predict()")

            scores.append(metric(y_test, y_predict))

        mean_score = np.mean(scores) 
        std_score = np.std(scores)


        return mean_score, std_score

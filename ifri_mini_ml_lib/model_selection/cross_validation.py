import numpy as np 
from collections import defaultdict

def k_fold_cross_validation (model, X, y, metric, stratified, k = 5):
        """
    Description:
        Implements k-Fold Cross-Validation, a technique for assessing the performance 
        of a machine learning model by dividing the dataset into k subsets (folds), 
        training the model on k-1 folds and testing it on the remaining one. This 
        process is repeated k times, each time with a different test fold. Optionally, 
        stratification ensures class balance across folds.

    Args:
        model: A machine learning model implementing .set_params(), .fit(), and .predict() methods.
        X: Input features (list or NumPy array) of shape (n_samples, n_features).
        y: Target values (list or NumPy array) of shape (n_samples,).
        metric: A scoring function to evaluate the model’s performance (e.g., accuracy, f1_score, mse).
        stratified: Boolean. If True, performs stratified k-fold (preserving label proportions).
        k: Integer, number of folds (default is 5). Must be between 2 and the number of samples.

    Returns:
        A tuple containing:
            - mean_score (float): The average metric score across all folds.
            - std_score (float): The standard deviation of the metric scores across folds.

    Example:
        >>> from ifri_mini_lib.supervised.classification import LogisticRegression
        >>> from ifri_mini_lib.metrics import accuracy_score
        >>> X = [[1], [2], [3], [4], [5]]
        >>> y = [0, 0, 1, 1, 1]
        >>> model = LogisticRegression()
        >>> k_fold_cross_validation(model, X, y, accuracy_score, stratified=True, k=3)
        (0.83, 0.12)  # Example output
        """
        
        X, y = np.array(X), np.array(y) 
        n_samples = len(X)

        #Gestion des erreurs sur le nbre de folds
        if k > n_samples or k < 2:
            raise ValueError("Number of folds must be between 2 and the number of samples.")
    
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

        #Cross Validation
        scores = []

        for test_indices in folds:
            #separons à present chaque fold en train/test
            train_indices = np.setdiff1d(indices, test_indices, assume_unique=True) #operation vectorielle optimisée

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            if not (hasattr(model, "fit") and hasattr(model, "predict")):
                raise ValueError("The model must implement both .fit() et .predict() methods")
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)

            scores.append(metric(y_test, y_predict))

        mean_score = np.mean(scores) 
        std_score = np.std(scores)


        return mean_score, std_score
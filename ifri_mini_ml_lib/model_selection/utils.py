def clone(estimator): 
    """
    Description:
        Clones a ifri_mini_lib estimator by creating a deep copy of its initial parameters 
        without copying any learned data (i.e., trained weights, fitted attributes, etc.).
        This is useful for reusing a model structure in cross-validation or hyperparameter tuning.

    Args:
        estimator: An object that implements ifri_mini_lib's `get_params()` and `set_params()` methods. 
                   Typically, this is any ifri_mini_lib-compatible estimator or pipeline.

    Returns:
        new_estimator: A fresh, unfitted instance of the same estimator class, 
                       initialized with the same parameters as the original.

    Example:
        >>> from ifri_mini_lib.supervised.classification import DecisionTreeClassifier
        >>> model = DecisionTreeClassifier(max_depth=3)
        >>> cloned_model = clone(model)
        >>> cloned_model is model
        False
        >>> cloned_model.get_params() == model.get_params()
        True
    """

    # Vérification que l'estimateur est compatible
    if not hasattr(estimator, 'get_params') or not hasattr(estimator, 'set_params'):
        raise ValueError("L'estimateur doit implémenter get_params() et set_params()")
    
    # 1. Récupère les paramètres initiaux
    params = estimator.get_params(deep=False)
    
    # 2. Crée une nouvelle instance
    # On utilise la classe d'origine pour instancier
    new_estimator = estimator.__class__()
    
    # 3. Applique les paramètres
    new_estimator.set_params(**params)
    
    return new_estimator

class BaseEstimatorMinimal:
    ''' A Base Estimator which implements minimalist methods get_params() and set_params()'''
    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
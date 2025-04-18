def clone(estimator):
    """
    Clone un estimateur scikit-learn en créant une copie profonde de ses paramètres
    sans copier ses données apprenées.
    
    Parameters
    ----------
    estimator : object
        L'estimateur à cloner (doit avoir get_params() et set_params())
        
    Returns
    -------
    new_estimator : object
        Une nouvelle instance non entraînée du même type
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
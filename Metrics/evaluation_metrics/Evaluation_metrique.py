def confusion_matrix(y_true, y_pred, classes=None):
    """
    Calcule la matrice de confusion.

    Args:
        y_true (list[int]): Liste des étiquettes réelles (valeurs observées).
        y_pred (list[int]): Liste des étiquettes prédites par le modèle.
        classes (list[int], optionnel): Liste des classes possibles. 
                                        Si None, les classes sont déduites des données.

    Returns:
        dict: Matrice de confusion sous forme de dictionnaire, 
              avec {classe_réelle: {classe_prédite: count}}.
              Par exemple, {0: {0: 50, 1: 10}, 1: {0: 5, 1: 100}}.
    """
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))  # Déduit les classes uniques présentes dans les deux listes.
    
    # Initialisation de la matrice de confusion avec des zéros.
    matrix = {true_class: {pred_class: 0 for pred_class in classes} 
              for true_class in classes}
    
    # Remplissage de la matrice de confusion en comptant les occurrences.
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    
    return matrix


def accuracy(y_true, y_pred):
    """
    Calcule l'accuracy (exactitude) du modèle.

    Args:
        y_true (list[int]): Liste des étiquettes réelles.
        y_pred (list[int]): Liste des étiquettes prédites.

    Returns:
        float: La précision, c'est-à-dire le rapport entre le nombre de prédictions correctes et le total des prédictions.
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)  # Nombre de prédictions correctes.
    return correct / len(y_true)  # Calcul de l'accuracy


def precision(y_true, y_pred, positive_class=1):
    """
    Calcule la précision pour une classe donnée.

    Args:
        y_true (list[int]): Liste des étiquettes réelles.
        y_pred (list[int]): Liste des étiquettes prédites.
        positive_class (int, optionnel): Classe considérée comme positive (par défaut 1).

    Returns:
        float: Valeur de la précision, c'est-à-dire la proportion de vrais positifs parmi tous les positifs prédits.
    """
    tp = fp = 0  # Initialisation des vrais positifs (tp) et faux positifs (fp)
    
    # Comptage des vrais et faux positifs
    for true, pred in zip(y_true, y_pred):
        if pred == positive_class:
            if true == positive_class:
                tp += 1  # Vrai positif
            else:
                fp += 1  # Faux positif
    
    return tp / (tp + fp) if (tp + fp) > 0 else 0  # Retour de la précision, ou 0 si il n'y a pas de faux positifs.


def recall(y_true, y_pred, positive_class=1):
    """
    Calcule le rappel (sensibilité) pour une classe donnée.

    Args:
        y_true (list[int]): Liste des étiquettes réelles.
        y_pred (list[int]): Liste des étiquettes prédites.
        positive_class (int, optionnel): Classe considérée comme positive (par défaut 1).

    Returns:
        float: Valeur du rappel, c'est-à-dire la proportion de vrais positifs parmi tous les vrais cas positifs.
    """
    tp = fn = 0  # Initialisation des vrais positifs (tp) et faux négatifs (fn)
    
    # Comptage des vrais positifs et faux négatifs
    for true, pred in zip(y_true, y_pred):
        if pred == positive_class and true == positive_class:
            tp += 1  # Vrai positif
        elif true == positive_class:
            fn += 1  # Faux négatif
    
    return tp / (tp + fn) if (tp + fn) > 0 else 0  # Retour du rappel, ou 0 si il n'y a pas de vrais cas positifs.


def f1_score(y_true, y_pred, positive_class=1):
    """
    Calcule le F1-score pour une classe donnée.

    Args:
        y_true (list[int]): Liste des étiquettes réelles.
        y_pred (list[int]): Liste des étiquettes prédites.
        positive_class (int, optionnel): Classe considérée comme positive (par défaut 1).

    Returns:
        float: Valeur du F1-score, qui est la moyenne harmonique de la précision et du rappel.
    """
    prec = precision(y_true, y_pred, positive_class)  # Calcul de la précision
    rec = recall(y_true, y_pred, positive_class)  # Calcul du rappel
    
    # Calcul du F1-score (moyenne harmonique)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

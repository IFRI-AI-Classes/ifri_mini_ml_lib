def confusion_matrix(y_true, y_pred, classes=None):
    """
    Calcule la matrice de confusion.
    
    Args:
        y_true: List[int] - Étiquettes réelles
        y_pred: List[int] - Étiquettes prédites
        classes: List[int] - Liste des classes (si None, déduit des données)
    
    Returns:
        dict: Matrice de confusion sous forme de dictionnaire
              {classe_réelle: {classe_prédite: count}}
    """
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))
    
    # Initialisation de la matrice
    matrix = {true_class: {pred_class: 0 for pred_class in classes} 
              for true_class in classes}
    
    # Remplissage de la matrice
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1
    
    return matrix


def accuracy(y_true, y_pred):
    """
    Calcule l'accuracy (exactitude).
    
    Args:
        y_true: List[int] - Étiquettes réelles
        y_pred: List[int] - Étiquettes prédites
    
    Returns:
        float: Accuracy entre 0 et 1
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)



def precision(y_true, y_pred, positive_class=1):
    """
    Calcule la précision pour une classe donnée.
    
    Args:
        y_true: List[int] - Étiquettes réelles
        y_pred: List[int] - Étiquettes prédites
        positive_class: int - Classe considérée comme positive
    
    Returns:
        float: Valeur de la précision
    """
    tp = fp = 0
    
    for true, pred in zip(y_true, y_pred):
        if pred == positive_class:
            if true == positive_class:
                tp += 1
            else:
                fp += 1
    
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred, positive_class=1):
    """
    Calcule le rappel pour une classe donnée.
    
    Args:
        y_true: List[int] - Étiquettes réelles
        y_pred: List[int] - Étiquettes prédites
        positive_class: int - Classe considérée comme positive
    
    Returns:
        float: Valeur du rappel
    """
    tp = fn = 0
    
    for true, pred in zip(y_true, y_pred):
        if pred == positive_class and true == positive_class:
            tp += 1
        elif true == positive_class:
            fn += 1
    
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(y_true, y_pred, positive_class=1):
    """
    Calcule le F1-score pour une classe donnée.
    
    Args:
        y_true: List[int] - Étiquettes réelles
        y_pred: List[int] - Étiquettes prédites
        positive_class: int - Classe considérée comme positive
    
    Returns:
        float: Valeur du F1-score
    """
    prec = precision(y_true, y_pred, positive_class)
    rec = recall(y_true, y_pred, positive_class)
    
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
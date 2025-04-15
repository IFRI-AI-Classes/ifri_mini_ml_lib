import numpy as np

def euclidean_distance(x1, x2):
    """
    Calcule la distance euclidienne entre deux vecteurs.

    Paramètres:
    - x1 (numpy.ndarray): Premier vecteur.
    - x2 (numpy.ndarray): Second vecteur.

    Retour:
    - float: La distance euclidienne entre x1 et x2.
    """
    return np.sqrt(np.sum((x1 - x2)**2))

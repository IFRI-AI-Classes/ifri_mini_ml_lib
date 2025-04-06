"""
Module contenant des fonctions pour calculer les métriques d'évaluation 
des règles d'association.
"""
from typing import Set, List, Dict, Union, FrozenSet
import warnings


def support(itemset: Union[Set, FrozenSet], transactions: List[Set]) -> float:
    """
    Calcule le support d'un itemset.
    
    Le support est la proportion de transactions qui contiennent l'itemset.
    
    Args:
        itemset: Ensemble d'items
        transactions: Liste des transactions
        
    Returns:
        float: Valeur du support entre 0 et 1
    """
    if not transactions:
        return 0
        
    return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)


def confidence(antecedent: Union[Set, FrozenSet], 
               consequent: Union[Set, FrozenSet], 
               transactions: List[Set]) -> float:
    """
    Calcule la confiance d'une règle d'association (X → Y).
    
    La confiance est la probabilité conditionnelle du conséquent sachant l'antécédent:
    conf(X → Y) = supp(X ∪ Y) / supp(X)
    
    Args:
        antecedent: Partie gauche de la règle (X)
        consequent: Partie droite de la règle (Y)
        transactions: Liste des transactions
        
    Returns:
        float: Valeur de confiance entre 0 et 1
    """
    if not transactions:
        return 0
    
    supp_ant = support(antecedent, transactions)
    
    if supp_ant == 0:
        warnings.warn("Support de l'antécédent est 0, confiance indéfinie (retourne 0)")
        return 0
    
    supp_total = support(antecedent.union(consequent), transactions)
    
    return supp_total / supp_ant


def lift(antecedent: Union[Set, FrozenSet], 
         consequent: Union[Set, FrozenSet], 
         transactions: List[Set]) -> float:
    """
    Calcule le lift d'une règle d'association (X → Y).
    
    Le lift mesure l'indépendance entre l'antécédent et le conséquent:
    lift(X → Y) = conf(X → Y) / supp(Y)
    
    Interprétation:
    - lift > 1: Association positive (X favorise Y)
    - lift = 1: Indépendance (X n'influence pas Y)
    - lift < 1: Association négative (X défavorise Y)
    
    Args:
        antecedent: Partie gauche de la règle (X)
        consequent: Partie droite de la règle (Y)
        transactions: Liste des transactions
        
    Returns:
        float: Valeur du lift (≥ 0)
    """
    if not transactions:
        return 0
    
    # Calculer la confiance
    conf_value = confidence(antecedent, consequent, transactions)
    
    # Calculer le support du conséquent
    supp_cons = support(consequent, transactions)
    
    if supp_cons == 0:
        warnings.warn("Support du conséquent est 0, lift indéfini (retourne 0)")
        return 0
    
    return conf_value / supp_cons


def conviction(antecedent: Union[Set, FrozenSet], 
               consequent: Union[Set, FrozenSet], 
               transactions: List[Set]) -> float:
    """
    Calcule la conviction d'une règle d'association (X → Y).
    
    La conviction mesure l'implication de la règle:
    conviction(X → Y) = (1 - supp(Y)) / (1 - conf(X → Y))
    
    Interprétation:
    - conviction > 1: X implique Y avec une force proportionnelle
    - conviction = 1: X et Y sont indépendants
    - conviction < 1: X implique ¬Y (la négation de Y)
    - conviction = ∞: Implication parfaite (confiance = 1)
    
    Args:
        antecedent: Partie gauche de la règle (X)
        consequent: Partie droite de la règle (Y)
        transactions: Liste des transactions
        
    Returns:
        float: Valeur de conviction (peut être infinie)
    """
    if not transactions:
        return 0
    
    # Calculer la confiance
    conf_value = confidence(antecedent, consequent, transactions)
    
    # Si la confiance est 1, la conviction est infinie
    if conf_value == 1:
        return float('inf')
    
    # Calculer le support du conséquent
    supp_cons = support(consequent, transactions)
    
    return (1 - supp_cons) / (1 - conf_value) if conf_value < 1 else float('inf')


def leverage(antecedent: Union[Set, FrozenSet], 
             consequent: Union[Set, FrozenSet], 
             transactions: List[Set]) -> float:
    """
    Calcule le leverage d'une règle d'association (X → Y).
    
    Le leverage mesure la différence entre la probabilité observée 
    de cooccurrence et celle attendue sous indépendance:
    leverage(X → Y) = supp(X ∪ Y) - (supp(X) * supp(Y))
    
    Interprétation:
    - leverage > 0: Association positive 
    - leverage = 0: Indépendance
    - leverage < 0: Association négative
    
    Args:
        antecedent: Partie gauche de la règle (X)
        consequent: Partie droite de la règle (Y)
        transactions: Liste des transactions
        
    Returns:
        float: Valeur du leverage entre -0.25 et 0.25
    """
    if not transactions:
        return 0
    
    # Calculer le support de l'union
    supp_union = support(antecedent.union(consequent), transactions)
    
    # Calculer les supports individuels
    supp_ant = support(antecedent, transactions)
    supp_cons = support(consequent, transactions)
    
    # Calculer le leverage
    return supp_union - (supp_ant * supp_cons)


def jaccard(antecedent: Union[Set, FrozenSet], 
            consequent: Union[Set, FrozenSet], 
            transactions: List[Set]) -> float:
    """
    Calcule le coefficient de Jaccard pour une règle d'association (X → Y).
    
    Le coefficient de Jaccard mesure la similarité entre les ensembles:
    jaccard(X, Y) = supp(X ∪ Y) / (supp(X) + supp(Y) - supp(X ∪ Y))
    
    Args:
        antecedent: Partie gauche de la règle (X)
        consequent: Partie droite de la règle (Y)
        transactions: Liste des transactions
        
    Returns:
        float: Valeur du coefficient de Jaccard entre 0 et 1
    """
    if not transactions:
        return 0
    
    # Calculer les supports
    supp_ant = support(antecedent, transactions)
    supp_cons = support(consequent, transactions)
    supp_union = support(antecedent.union(consequent), transactions)
    
    # Éviter la division par zéro
    denominator = supp_ant + supp_cons - supp_union
    if denominator == 0:
        return 0
    
    return supp_union / denominator


def evaluate_rule(antecedent: Union[Set, FrozenSet], 
                 consequent: Union[Set, FrozenSet], 
                 transactions: List[Set]) -> Dict[str, float]:
    """
    Évalue une règle d'association selon plusieurs métriques.
    
    Args:
        antecedent: Partie gauche de la règle (X)
        consequent: Partie droite de la règle (Y)
        transactions: Liste des transactions
        
    Returns:
        Dict: Dictionnaire contenant les différentes métriques calculées
    """
    return {
        'support': support(antecedent.union(consequent), transactions),
        'confidence': confidence(antecedent, consequent, transactions),
        'lift': lift(antecedent, consequent, transactions),
        'conviction': conviction(antecedent, consequent, transactions),
        'leverage': leverage(antecedent, consequent, transactions),
        'jaccard': jaccard(antecedent, consequent, transactions)
    }
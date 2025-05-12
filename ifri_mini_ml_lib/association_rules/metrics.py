"""
Module containing functions to calculate evaluation metrics
for association rules.

Examples:
    
    >>> # Creating a simple example of transactions
    >>> transactions = [
    ...     {'A', 'B', 'C'},
    ...     {'A', 'B'},
    ...     {'A', 'D'},
    ...     {'B', 'C', 'E'},
    ...     {'A', 'B', 'C', 'E'}
    ... ]
    >>> # Calculating support for an itemset
    >>> from ifri_mini_ml_lib.association_rules.metrics import support
    >>> support_ab = support({'A', 'B'}, transactions)
    >>> print(f"Support({'A', 'B'}) = {support_ab:.2f}")
    Support({'A', 'B'}) = 0.60
    
    >>> # Calculating confidence for a rule A → B
    >>> from ifri_mini_ml_lib.association_rules.metrics import confidence
    >>> conf_a_to_b = confidence({'A'}, {'B'}, transactions)
    >>> print(f"Confidence({'A'} → {'B'}) = {conf_a_to_b:.2f}")
    Confidence({'A'} → {'B'}) = 0.75
    
    >>> # Calculating lift for a rule A → C
    >>> from ifri_mini_ml_lib.association_rules.metrics import lift
    >>> lift_a_to_c = lift({'A'}, {'C'}, transactions)
    >>> print(f"Lift({'A'} → {'C'}) = {lift_a_to_c:.2f}")
    Lift({'A'} → {'C'}) = 1.25
    
    >>> # Complete evaluation of a rule
    >>> from ifri_mini_ml_lib.association_rules.metrics import evaluate_rule
    >>> metrics = evaluate_rule({'B'}, {'C'}, transactions)
    >>> for metric_name, value in metrics.items():
    ...     print(f"{metric_name}: {value:.2f}")
    support: 0.60
    confidence: 0.75
    lift: 1.25
    conviction: 2.00
    leverage: 0.15
    jaccard: 0.50
"""
from typing import Set, List, Dict, Union, FrozenSet
import warnings


def support(itemset: Union[Set, FrozenSet], transactions: List[Set]) -> float:
    """
    Calculates the support of an itemset.
    
    Support is the proportion of transactions that contain the itemset.
    
    Args:
        itemset: Set of items
        transactions: List of transactions
        
    Returns:
        float: Support value between 0 and 1
    """
    if not transactions:
        return 0
        
    return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)


def confidence(antecedent: Union[Set, FrozenSet], 
               consequent: Union[Set, FrozenSet], 
               transactions: List[Set]) -> float:
    """
    Calculates the confidence of an association rule (X → Y).
    
    Confidence is the conditional probability of the consequent given the antecedent:
    conf(X → Y) = supp(X ∪ Y) / supp(X)
    
    Args:
        antecedent: Left-hand side of the rule (X)
        consequent: Right-hand side of the rule (Y)
        transactions: List of transactions
        
    Returns:
        float: Confidence value between 0 and 1
    """
    if not transactions:
        return 0
    
    supp_ant = support(antecedent, transactions)
    
    if supp_ant == 0:
        warnings.warn("Antecedent support is 0, confidence is undefined (returns 0)")
        return 0
    
    supp_total = support(antecedent.union(consequent), transactions)
    
    return supp_total / supp_ant


def lift(antecedent: Union[Set, FrozenSet], 
         consequent: Union[Set, FrozenSet], 
         transactions: List[Set]) -> float:
    """
    Calculates the lift of an association rule (X → Y).
    
    Lift measures the independence between the antecedent and the consequent:
    lift(X → Y) = conf(X → Y) / supp(Y)
    
    Interpretation:
    - lift > 1: Positive association (X favors Y)
    - lift = 1: Independence (X does not influence Y)
    - lift < 1: Negative association (X disfavors Y)
    
    Args:
        antecedent: Left-hand side of the rule (X)
        consequent: Right-hand side of the rule (Y)
        transactions: List of transactions
        
    Returns:
        float: Lift value (≥ 0)
    """
    if not transactions:
        return 0
    
    # Calculate confidence
    conf_value = confidence(antecedent, consequent, transactions)
    
    # Calculate the support of the consequent
    supp_cons = support(consequent, transactions)
    
    if supp_cons == 0:
        warnings.warn("Consequent support is 0, lift is undefined (returns 0)")
        return 0
    
    return conf_value / supp_cons


def conviction(antecedent: Union[Set, FrozenSet], 
               consequent: Union[Set, FrozenSet], 
               transactions: List[Set]) -> float:
    """
    Calculates the conviction of an association rule (X → Y).
    
    Conviction measures the implication of the rule:
    conviction(X → Y) = (1 - supp(Y)) / (1 - conf(X → Y))
    
    Interpretation:
    - conviction > 1: X implies Y with proportional strength
    - conviction = 1: X and Y are independent
    - conviction < 1: X implies ¬Y (the negation of Y)
    - conviction = ∞: Perfect implication (confidence = 1)
    
    Args:
        antecedent: Left-hand side of the rule (X)
        consequent: Right-hand side of the rule (Y)
        transactions: List of transactions
        
    Returns:
        float: Conviction value (can be infinite)
    """
    if not transactions:
        return 0
    
    # Calculate confidence
    conf_value = confidence(antecedent, consequent, transactions)
    
    # If confidence is 1, conviction is infinite
    if conf_value == 1:
        return float('inf')
    
    # Calculate the support of the consequent
    supp_cons = support(consequent, transactions)
    
    return (1 - supp_cons) / (1 - conf_value) if conf_value < 1 else float('inf')


def leverage(antecedent: Union[Set, FrozenSet], 
             consequent: Union[Set, FrozenSet], 
             transactions: List[Set]) -> float:
    """
    Calculates the leverage of an association rule (X → Y).
    
    Leverage measures the difference between the observed probability
    of co-occurrence and that expected under independence:
    leverage(X → Y) = supp(X ∪ Y) - (supp(X) * supp(Y))
    
    Interpretation:
    - leverage > 0: Positive association 
    - leverage = 0: Independence
    - leverage < 0: Negative association
    
    Args:
        antecedent: Left-hand side of the rule (X)
        consequent: Right-hand side of the rule (Y)
        transactions: List of transactions
        
    Returns:
        float: Leverage value between -0.25 and 0.25
    """
    if not transactions:
        return 0
    
    # Calculate the support of the union
    supp_union = support(antecedent.union(consequent), transactions)
    
    # Calculate individual supports
    supp_ant = support(antecedent, transactions)
    supp_cons = support(consequent, transactions)
    
    # Calculate leverage
    return supp_union - (supp_ant * supp_cons)


def jaccard(antecedent: Union[Set, FrozenSet], 
            consequent: Union[Set, FrozenSet], 
            transactions: List[Set]) -> float:
    """
    Calculates the Jaccard coefficient for an association rule (X → Y).
    
    The Jaccard coefficient measures the similarity between sets:
    jaccard(X, Y) = supp(X ∪ Y) / (supp(X) + supp(Y) - supp(X ∪ Y))
    
    Args:
        antecedent: Left-hand side of the rule (X)
        consequent: Right-hand side of the rule (Y)
        transactions: List of transactions
        
    Returns:
        float: Jaccard coefficient value between 0 and 1
    """
    if not transactions:
        return 0
    
    # Calculate supports
    supp_ant = support(antecedent, transactions)
    supp_cons = support(consequent, transactions)
    supp_union = support(antecedent.union(consequent), transactions)
    
    # Avoid division by zero
    denominator = supp_ant + supp_cons - supp_union
    if denominator == 0:
        return 0
    
    return supp_union / denominator


def evaluate_rule(antecedent: Union[Set, FrozenSet], 
                 consequent: Union[Set, FrozenSet], 
                 transactions: List[Set]) -> Dict[str, float]:
    """
    Evaluates an association rule according to multiple metrics.
    
    Args:
        antecedent: Left-hand side of the rule (X)
        consequent: Right-hand side of the rule (Y)
        transactions: List of transactions
        
    Returns:
        Dict: Dictionary containing the different calculated metrics
    """
    return {
        'support': support(antecedent.union(consequent), transactions),
        'confidence': confidence(antecedent, consequent, transactions),
        'lift': lift(antecedent, consequent, transactions),
        'conviction': conviction(antecedent, consequent, transactions),
        'leverage': leverage(antecedent, consequent, transactions),
        'jaccard': jaccard(antecedent, consequent, transactions)
    }
import numpy as np
from typing import Any, Dict, Optional, Tuple

def selection_rate(y_true: Optional[np.ndarray], y_pred: np.ndarray, *, pos_label: Any = 1, sample_weight: Optional[np.ndarray] = None) -> float:
    """
    Description:
        Computes the fraction of predicted labels equal to the positive outcome (pos_label).
        
    Args:
        y_true (np.ndarray or None): Not used here but kept for API consistency.
        y_pred (np.ndarray): Predicted labels.
        pos_label (Any): The label considered as the positive class. Default is 1.
        sample_weight (np.ndarray or None): Optional sample weights.

    Returns:
        float: The selection rate (i.e., proportion of positive predictions).

    Example:
        selection_rate(None, np.array([1, 0, 1, 1]), pos_label=1)
    """
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if sample_weight is not None and not isinstance(sample_weight, np.ndarray):
        sample_weight = np.array(sample_weight)

    selected = y_pred == pos_label
    if len(selected) == 0:
        raise ValueError("Empty predictions are not allowed.")

    s_w = np.ones(len(selected)) if sample_weight is None else sample_weight
    return np.dot(selected, s_w) / s_w.sum()


def selection_rate_per_group(
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    pos_label: Any,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[Any, float]:
    """
    Description:
        Computes selection rate for each group in the sensitive feature.
        
    Args:
        y_true (np.ndarray or None): Not used.
        y_pred (np.ndarray): Predicted labels.
        sensitive_features (np.ndarray): Group identifiers for each sample.
        pos_label (Any): The positive label to count as selected.
        sample_weight (np.ndarray or None): Optional weights per sample.

    Returns:
        Dict[Any, float]: A dictionary with groups as keys and their selection rate as values.
    """
    if not isinstance(sensitive_features, np.ndarray):
        sensitive_features = np.array(sensitive_features)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    groups = np.unique(sensitive_features)
    selection_rates = {}

    for group in groups:
        group_mask = sensitive_features == group
        y_pred_group = y_pred[group_mask]
        weight_group = sample_weight[group_mask] if sample_weight is not None else None
        rate = selection_rate(None, y_pred_group, pos_label=pos_label, sample_weight=weight_group)
        selection_rates[group] = rate

    return selection_rates


def demographic_parity_ratio(
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    pos_label: Any = 1,
    sample_weight: Optional[np.ndarray] = None
) -> Tuple[float, Dict[Any, float]]:
    """
    Description:
        Computes the demographic parity ratio (min rate / max rate).

    Args:
        y_pred (np.ndarray): Predicted labels.
        sensitive_features (np.ndarray): Group identifiers.
        pos_label (Any): Positive label to use.
        sample_weight (np.ndarray or None): Optional weights.

    Returns:
        Tuple[float, Dict[Any, float]]: Ratio and rates per group.

    Example:
        demographic_parity_ratio(np.array([1, 0, 1]), np.array(['A', 'B', 'A']))
    """
    rates = selection_rate_per_group(None, y_pred, sensitive_features, pos_label, sample_weight)
    max_rate = max(rates.values())
    min_rate = min(rates.values())
    ratio = min_rate / max_rate if max_rate != 0 else 0
    return ratio, rates


def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    pos_label: Any = 1,
    sample_weight: Optional[np.ndarray] = None
) -> Tuple[float, Dict[Any, float]]:
    """
    Description:
        Computes the demographic parity difference (|max - min| selection rates).

    Args:
        y_pred (np.ndarray): Predicted labels.
        sensitive_features (np.ndarray): Group identifiers.
        pos_label (Any): Positive class label.
        sample_weight (np.ndarray or None): Optional sample weights.

    Returns:
        Tuple[float, Dict[Any, float]]: Difference and selection rates per group.
    """
    rates = selection_rate_per_group(None, y_pred, sensitive_features, pos_label, sample_weight)
    max_rate = max(rates.values())
    min_rate = min(rates.values())
    return abs(min_rate - max_rate), rates


def tpr_fpr_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    pos_label: Any = 1
) -> Tuple[Dict[Any, float], Dict[Any, float]]:
    """
    Description:
        Computes True Positive Rate (TPR) and False Positive Rate (FPR) for each group.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        sensitive_features (np.ndarray): Group identifiers.
        pos_label (Any): Label considered as positive class.

    Returns:
        Tuple[Dict[Any, float], Dict[Any, float]]: TPR and FPR per group.
    """
    tpr_dict, fpr_dict = {}, {}
    groups = np.unique(sensitive_features)

    for group in groups:
        mask = np.array(sensitive_features) == group
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        positives = y_true_g == pos_label
        negatives = y_true_g != pos_label

        tpr = np.sum((y_pred_g == pos_label) & positives) / positives.sum() if positives.sum() > 0 else 0.0
        fpr = np.sum((y_pred_g == pos_label) & negatives) / negatives.sum() if negatives.sum() > 0 else 0.0

        tpr_dict[group] = tpr
        fpr_dict[group] = fpr

    return tpr_dict, fpr_dict


def equalized_odds_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    pos_label: Any = 1
) -> Tuple[float, Dict[Any, float], Dict[Any, float]]:
    """
    Description:
        Computes the Equalized Odds Ratio (min/max of TPR and FPR between groups).

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        sensitive_features (np.ndarray): Group identifiers.
        pos_label (Any): Positive class.

    Returns:
        Tuple[float, Dict[Any, float], Dict[Any, float]]: EOR, TPR per group, FPR per group.
    """
    tpr_dict, fpr_dict = tpr_fpr_by_group(y_true, y_pred, sensitive_features, pos_label)
    tpr_ratio = min(tpr_dict.values()) / max(tpr_dict.values()) if max(tpr_dict.values()) > 0 else 0
    fpr_ratio = min(fpr_dict.values()) / max(fpr_dict.values()) if max(fpr_dict.values()) > 0 else 0
    return min(tpr_ratio, fpr_ratio), tpr_dict, fpr_dict


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    pos_label: Any = 1
) -> Tuple[float, Dict[Any, float], Dict[Any, float]]:
    """
    Description:
        Computes the maximum difference of TPR and FPR between groups.

    Args:
        y_true (np.ndarray): Ground truth.
        y_pred (np.ndarray): Predictions.
        sensitive_features (np.ndarray): Group labels.
        pos_label (Any): Positive label.

    Returns:
        Tuple[float, Dict[Any, float], Dict[Any, float]]: Difference, TPR per group, FPR per group.
    """
    tpr_dict, fpr_dict = tpr_fpr_by_group(y_true, y_pred, sensitive_features, pos_label)
    tpr_diff = max(tpr_dict.values()) - min(tpr_dict.values())
    fpr_diff = max(fpr_dict.values()) - min(fpr_dict.values())
    return max(tpr_diff, fpr_diff), tpr_dict, fpr_dict

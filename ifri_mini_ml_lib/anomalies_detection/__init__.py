"""
Anomalies Detection Module.

This module provides various methods for detecting outliers and anomalies in datasets,
including statistical methods and algorithmic approaches.

Available methods:
    - IQR: Robust anomaly detection using the Interquartile Range.
    - zscore_detection: Classic Z-score method (mean/std based).
    - modified_zscore_detection: Robust Z-score method (median/MAD based).
    - IsolationForest: Tree-based anomaly detection (Implementation pending).
"""

from .z_score import zscore_detection, modified_zscore_detection, summary_anomalies
from .iqr import IQR

# IsolationForest implementation is pending in isolation.py
try:
    from .isolation import IsolationForest
except (ImportError, AttributeError):
    IsolationForest = None

__all__ = [
    "zscore_detection",
    "modified_zscore_detection",
    "summary_anomalies",
    "IQR",
    "IsolationForest",
]

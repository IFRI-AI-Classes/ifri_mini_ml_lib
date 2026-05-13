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
from .iqr import IQR
from .isolation import IsolationForest
from .z_score import Z_Score


__all__ = [
    "IQR",
    "IsolationForest",
    "Z_Score"
]


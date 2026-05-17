"""
Anomalies Detection Module.

This module provides various methods for detecting outliers and anomalies in datasets,
including statistical methods and algorithmic approaches.

Available methods:
    - IQR: Robust anomaly detection using the Interquartile Range.
    - modified_zscore_detection: Robust Z-score method (median/MAD based).
    - IsolationForest: Tree-based anomaly detection.
"""
from .iqr import IQR
from .isolation_forest import IsolationForest
from .z_score import ZScoreDetector, modified_zscore_detection


__all__ = [
    "IQR",
    "IsolationForest",
    "ZScoreDetector",
    "modified_zscore_detection",
]
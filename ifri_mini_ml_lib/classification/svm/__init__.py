"""
Support Vector Machine (SVM) Package.

Provides:
    - BaseSVM: Abstract base class for all SVM models
    - LinearSVM: Linear SVM with Pegasos optimization
    - SVMClassifier: Binary RBF SVM with SMO optimization
    - SVMClassifierOvO: Multiclass RBF SVM with One-vs-One strategy
    - rbf_kernel: RBF kernel computation function
    - smo: SMO optimization algorithm
"""

from .svm import (
    BaseSVM,
    LinearSVM,
    SVMClassifier,
    SVMClassifierOvO,
    rbf_kernel,
    smo,
)

__all__ = [
    "BaseSVM",
    "LinearSVM",
    "SVMClassifier",
    "SVMClassifierOvO",
    "rbf_kernel",
    "smo",
]

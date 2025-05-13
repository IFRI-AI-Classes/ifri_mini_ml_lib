"""
# IFRI Mini ML Library

A concise and educational implementation of machine learning algorithms.

## Features

- **Classification**: Decision Trees, KNN, Logistic Regression
- **Clustering**: K-Means, DBSCAN, Hierarchical
- **Regression**: Linear, Polynomial, SVR
- **Neural Networks**: MLP Classifier and Regressor
- **Preprocessing**: Scalers, Encoders, PCA, t-SNE
- **Model Selection**: Cross-validation, Grid Search, Bayesian Search
- **Association Rules**: Apriori, Eclat, FP-Growth
- **Metrics**: Accuracy, Precision, Recall, F1 Score, Silhouette Score, Inertia
"""
from . import classification
from . import clustering
from . import regression
from . import preprocessing
from . import model_selection
from . import neural_networks
from . import association_rules
from . import metrics

version = "0.1.1"
__all__ = [
    "classification",
    "clustering",
    "regression",
    "preprocessing",
    "model_selection",
    "neural_networks",
    "association_rules",
    "metrics",
]
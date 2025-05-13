# ifri_mini_ml_lib

[![PyPI version](https://img.shields.io/pypi/v/ifri-mini-ml-lib.svg)](https://pypi.org/project/ifri-mini-ml-lib/) ![Coverage](https://img.shields.io/badge/coverage-dynamic-lightgrey?style=flat&logo=codecov)

A lightweight, educational machine learning library reimplementing core algorithms from scratch, inspired by scikit-learn. Developed by IFRI AI students for the Concepts & Applications of Machine Learning course.

---

## Features

- Core machine learning algorithms for:
  - Classification (Decision Trees, KNN, Logistic Regression)
  - Regression (Linear, Polynomial, SVR)
  - Clustering (K-means, DBSCAN, Hierarchical)
  - Association Rules (Apriori, Eclat, FP-Growth)
  - Neural Networks (MLP)
- Model selection tools (Cross-validation, Grid Search, etc.)
- Preprocessing utilities (scalers, encoders, missing value handlers, etc.)
- Focus on transparency and understanding of ML model internals

## Installation

Install from PyPI:

```bash
pip install ifri-mini-ml-lib
```

Or install from source:

```bash
git clone https://github.com/IFRI-AI-Classes/ifri_mini_ml_lib.git
cd ifri_mini_ml_lib
pip install -e .
```

## Quick Start

Here's a simple example using the KNN classifier:

```python
from ifri_mini_ml_lib.classification import KNN

# Example data
data = [[0, 0], [1, 1], [0, 1], [1, 0]]
labels = [0, 1, 1, 0]

# Initialize and fit the model
knn = KNN(k=3)
knn.fit(data, labels)

# Predict
prediction = knn.predict([[0.9, 0.8]])
print(prediction)
```

## Documentation

Full documentation is available at: [ifri_mini_ml_lib.github.io](https://ifri-ai-classes.github.io/ifri_mini_ml_lib/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Thanks to the IFRI AI students and faculty who contributed to this project.

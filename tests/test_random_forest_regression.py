import numpy as np
from ifri_mini_ml_lib.regression.random_forest_regression import RandomForestRegressor
import pytest 


# Test model initialization
def test_model_initialization():

    model = RandomForestRegressor(
        n_estimators=5,
        max_depth=3,
        max_features=2
    )

    # Check parameters
    assert model.n_estimators == 5
    assert model.max_depth == 3
    assert model.max_features == 2

    # Trees list should be empty at start
    assert model.trees == []


# Test bootstrap sample dimensions
def test_bootstrap_sample_shape():

    model = RandomForestRegressor()

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([10, 20, 30, 40])

    # Generate bootstrap sample
    X_sample, y_sample = model._bootstrap_sample(X, y)

    # Shapes should remain the same
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape


# Test if trees are created during training
def test_fit_creates_trees():

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([2, 4, 6, 8])

    model = RandomForestRegressor(
        n_estimators=4,
        max_depth=2
    )

    # Train the model
    model.fit(X, y)

    # Check number of trained trees
    assert len(model.trees) == 4


# Test prediction output shape
def test_predict_output_shape():

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([2, 4, 6, 8])

    model = RandomForestRegressor(
        n_estimators=3,
        max_depth=2
    )

    model.fit(X, y)

    # Generate predictions
    predictions = model.predict(X)

    # Predictions should match target shape
    assert predictions.shape == y.shape


# Test prediction return type
def test_predict_returns_numpy_array():

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([2, 4, 6, 8])

    model = RandomForestRegressor()

    model.fit(X, y)

    predictions = model.predict(X)

    # Output should be a numpy array
    assert isinstance(predictions, np.ndarray)
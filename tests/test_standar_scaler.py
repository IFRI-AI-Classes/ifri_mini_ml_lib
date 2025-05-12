import numpy as np
import pandas as pd
import pytest
from ifri_mini_ml_lib.preprocessing.preparation import StandardScaler

def test_fit_and_transform_numpy_array():
    scaler = StandardScaler()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-7)

def test_fit_and_transform_dataframe():
    scaler = StandardScaler()
    df = pd.DataFrame({"A": [1, 3, 5], "B": [2, 4, 6]})
    df_scaled = scaler.fit_transform(df)

    assert isinstance(df_scaled, pd.DataFrame)
    assert np.allclose(df_scaled.mean().values, 0, atol=1e-7)
    assert np.allclose(df_scaled.std(ddof=0).values, 1, atol=1e-7)

def test_inverse_transform_numpy():
    scaler = StandardScaler()
    X = np.array([[10, 20], [30, 40], [50, 60]])
    X_scaled = scaler.fit_transform(X)
    X_restored = scaler.inverse_transform(X_scaled)

    assert np.allclose(X, X_restored)

def test_inverse_transform_dataframe():
    scaler = StandardScaler()
    df = pd.DataFrame({"X": [10, 20, 30], "Y": [5, 15, 25]})
    df_scaled = scaler.fit_transform(df)
    df_restored = scaler.inverse_transform(df_scaled)

    assert isinstance(df_restored, pd.DataFrame)
    assert np.allclose(df.values, df_restored.values)

def test_transform_without_fit_raises_error():
    scaler = StandardScaler()
    with pytest.raises(ValueError):
        scaler.transform([[1, 2], [3, 4]])

def test_inverse_transform_without_fit_raises_error():
    scaler = StandardScaler()
    with pytest.raises(ValueError):
        scaler.inverse_transform([[0, 0], [1, 1]])

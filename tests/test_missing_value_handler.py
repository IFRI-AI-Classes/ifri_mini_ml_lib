import numpy as np
import pandas as pd
import pytest
from ifri_mini_ml_lib.preprocessing.preparation import MissingValueHandler

# Fixtures
@pytest.fixture
def handler():
    return MissingValueHandler()

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, np.nan, 3, 4],
        'B': [5, 6, np.nan, 8],
        'C': [9, 10, 11, 12]
    })

# remove_missing
# remove_missing
def test_remove_missing_rows(handler, sample_df):
    result = handler.remove_missing(sample_df, threshold=0.75, axis=0)
    assert result.shape[0] == 2  # Two rows should remain in the resulting DataFrame

def test_remove_missing_columns(handler):
    df = pd.DataFrame({
        'A': [np.nan, np.nan, 3],
        'B': [1, 2, 3],
        'C': [np.nan, np.nan, np.nan]
    })
    result = handler.remove_missing(df, threshold=0.5, axis=1)
    assert 'C' not in result.columns

# impute_statistical
def test_impute_statistical_mean(handler, sample_df):
    result = handler.impute_statistical(sample_df.copy(), strategy="mean")
    assert not result.isnull().any().any()
    assert np.isclose(result.loc[1, 'A'], (1 + 3 + 4) / 3)

def test_impute_statistical_mode(handler):
    df = pd.DataFrame({
        'A': [1, 1, np.nan, 2]
    })
    result = handler.impute_statistical(df, strategy="mode")
    assert result.loc[2, 'A'] == 1
# impute_default
def test_impute_default(handler, sample_df):
    result = handler.impute_default(sample_df, value=-1)
    assert (result == -1).sum().sum() == 2

# impute_knn
def test_impute_knn(handler):
    df = pd.DataFrame({
        'X1': [1, 2, 3, 4, np.nan],
        'X2': [5, 6, 7, 8, 9]
    })
    result = handler.impute_knn(df, k=2, task='regression')
    assert not result.isnull().any().any()

# impute_regression
def test_impute_regression(handler):
    df = pd.DataFrame({
        'X1': [1, 2, 3, 4, 5],
        'X2': [2, 4, 6, 8, np.nan]
    })
    result = handler.impute_regression(df, target_col='X2')
    assert not result['X2'].isnull().any()

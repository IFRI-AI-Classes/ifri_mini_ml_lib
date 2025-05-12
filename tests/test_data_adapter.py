import pytest
import pandas as pd
import numpy as np
from ifri_mini_ml_lib.association_rules import DataAdapter

@pytest.fixture
def transactions():
    """Provide the fixed transactions list."""
    return [
        {'bread', 'milk', 'butter'},
        {'bread', 'jam', 'eggs'},
        {'milk', 'butter', 'cheese'},
        {'bread', 'milk', 'butter', 'cheese'},
        {'bread', 'jam', 'milk'}
    ]

def test_convert_from_dataframe_binary():
    """Test DataAdapter conversion from DataFrame in binary mode."""
    df = pd.DataFrame({
        'bread': [1, 1, 0, 1, 1],
        'milk': [1, 1, 1, 1, 1],
        'butter': [1, 0, 1, 1, 0],
        'jam': [0, 1, 0, 0, 1],
        'eggs': [0, 1, 0, 0, 0],
        'cheese': [0, 0, 1, 1, 0]
    })
    transactions = DataAdapter.convert_to_transactions(df, binary_mode=True)
    expected = [
        {'bread', 'milk', 'butter'},
        {'bread', 'milk', 'jam', 'eggs'},
        {'milk', 'butter', 'cheese'},
        {'bread', 'milk', 'butter', 'cheese'},
        {'bread', 'milk', 'jam'}
    ]
    assert len(transactions) == len(expected)
    assert all(t in expected for t in transactions)

def test_convert_from_dataframe_categorical():
    """Test DataAdapter conversion from DataFrame in categorical mode."""
    df = pd.DataFrame({
        'item1': ['bread', 'bread', 'milk', 'bread', 'bread'],
        'item2': ['milk', 'jam', 'butter', 'milk', 'jam'],
        'item3': ['butter', 'eggs', 'cheese', 'butter', 'milk']
    })
    transactions = DataAdapter.convert_to_transactions(df, binary_mode=False)
    expected = [
        {'item1_bread', 'item2_milk', 'item3_butter'},
        {'item1_bread', 'item2_jam', 'item3_eggs'},
        {'item1_milk', 'item2_butter', 'item3_cheese'},
        {'item1_bread', 'item2_milk', 'item3_butter'},
        {'item1_bread', 'item2_jam', 'item3_milk'}
    ]
    assert len(transactions) == len(expected)
    assert all(t in expected for t in transactions)

def test_convert_from_numpy_binary():
    """Test DataAdapter conversion from NumPy array in binary mode."""
    arr = np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0]
    ])
    transactions = DataAdapter.convert_to_transactions(arr, binary_mode=True)
    expected = [
        {'feature_0', 'feature_1', 'feature_2'},
        {'feature_0', 'feature_1', 'feature_3', 'feature_4'},
        {'feature_1', 'feature_2', 'feature_5'},
        {'feature_0', 'feature_1', 'feature_2', 'feature_5'},
        {'feature_0', 'feature_1', 'feature_3'}
    ]
    assert len(transactions) == len(expected)
    assert all(t in expected for t in transactions)

def test_convert_from_list(transactions):
    """Test DataAdapter conversion from list of sets."""
    result = DataAdapter.convert_to_transactions(transactions)
    assert len(result) == 5
    assert result == transactions

def test_invalid_input():
    """Test DataAdapter with invalid input."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        DataAdapter.convert_to_transactions("invalid_data")
    
    with pytest.raises(ValueError, match="Data cannot be empty"):
        DataAdapter.convert_to_transactions([])

def test_convert_from_dataframe_with_missing_values():
    """Test DataAdapter handling of missing values in DataFrame."""
    df = pd.DataFrame({
        'bread': [1, 1, 0, 1, None],
        'milk': [1, 1, 1, 1, 1],
        'butter': [1, 0, 1, 1, 0]
    })
    transactions = DataAdapter.convert_to_transactions(df, binary_mode=True)
    expected = [
        {'bread', 'milk', 'butter'},
        {'bread', 'milk'},
        {'milk', 'butter'},
        {'bread', 'milk', 'butter'},
        {'milk'}
    ]
    assert len(transactions) == len(expected)
    assert all(t in expected for t in transactions)
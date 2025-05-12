import pytest
import pandas as pd
import numpy as np
from ifri_mini_ml_lib.association_rules import DataAdapter

def test_convert_from_dataframe_binary():
    """Test DataAdapter conversion from DataFrame in binary mode."""
    df = pd.DataFrame({
        'item1': [1, 0, 1, 1, 0],
        'item2': [1, 1, 0, 1, 1],
        'item3': [0, 1, 1, 1, 0]
    })
    transactions = DataAdapter.convert_to_transactions(df, binary_mode=True)
    expected = [
        {'item1', 'item2'},
        {'item2', 'item3'},
        {'item1', 'item3'},
        {'item1', 'item2', 'item3'},
        {'item2'}
    ]
    assert len(transactions) == len(expected)
    assert all(t in expected for t in transactions)

def test_convert_from_dataframe_categorical():
    """Test DataAdapter conversion from DataFrame in categorical mode."""
    df = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'red', 'blue'],
        'size': ['large', 'medium', 'small', 'medium', 'large']
    })
    transactions = DataAdapter.convert_to_transactions(df, binary_mode=False)
    expected = [
        {'color_red', 'size_large'},
        {'color_blue', 'size_medium'},
        {'color_green', 'size_small'},
        {'color_red', 'size_medium'},
        {'color_blue', 'size_large'}
    ]
    assert len(transactions) == len(expected)
    assert all(t in expected for t in transactions)

def test_convert_from_numpy_binary():
    """Test DataAdapter conversion from NumPy array in binary mode."""
    arr = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ])
    transactions = DataAdapter.convert_to_transactions(arr, binary_mode=True)
    expected = [
        {'feature_0', 'feature_1'},
        {'feature_1', 'feature_2'},
        {'feature_0', 'feature_2'}
    ]
    assert len(transactions) == len(expected)
    assert all(t in expected for t in transactions)

def test_convert_from_list():
    """Test DataAdapter conversion from list of sets."""
    data = [
        {'bread', 'milk'},
        {'bread', 'diaper', 'beer'},
        {'milk', 'diaper', 'beer'}
    ]
    transactions = DataAdapter.convert_to_transactions(data)
    assert len(transactions) == 3
    assert transactions == data

def test_load_csv_to_transactions():
    """Test DataAdapter loading CSV to transactions."""
    transactions = DataAdapter.load_csv_to_transactions("tests/data/Market_Basket_Optimisation.csv")
    assert len(transactions) > 0
    assert all(isinstance(t, set) for t in transactions)
    assert all(all(isinstance(item, str) for item in t) for t in transactions)

def test_invalid_input():
    """Test DataAdapter with invalid input."""
    with pytest.raises(TypeError, match="Unsupported data type"):
        DataAdapter.convert_to_transactions("invalid_data")
    
    with pytest.raises(ValueError, match="Data cannot be empty"):
        DataAdapter.convert_to_transactions([])

def test_missing_csv_file():
    """Test DataAdapter with non-existent CSV file."""
    with pytest.raises(FileNotFoundError, match="File nonexistent.csv not found"):
        DataAdapter.load_csv_to_transactions("nonexistent.csv")
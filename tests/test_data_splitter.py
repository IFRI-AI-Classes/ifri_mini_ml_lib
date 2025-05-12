import pytest
import numpy as np
import pandas as pd
from ifri_mini_ml_lib.preprocessing.preparation import DataSplitter

@pytest.fixture
def sample_data():
    """Fixture pour créer des données de test communes."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))  
    return X, y

@pytest.fixture
def temporal_data():
    """Fixture pour créer des données temporelles."""
    dates = pd.date_range('2020-01-01', periods=100)
    X = pd.DataFrame({'feature': np.random.rand(100)}, index=dates)
    y = pd.Series(np.random.rand(100), index=dates)
    return X, y

def test_train_test_split_no_y(sample_data):
    X, _ = sample_data
    splitter = DataSplitter(seed=42)
    X_train, X_test = splitter.train_test_split(X)
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) / len(X) == pytest.approx(0.2, rel=0.05)

def test_train_test_split_with_y(sample_data):
    X, y = sample_data
    splitter = DataSplitter(seed=42)
    X_train, X_test, y_train, y_test = splitter.train_test_split(X, y)
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert set(X_train.index) == set(y_train.index)

def test_train_test_split_with_val(sample_data):
    X, y = sample_data
    splitter = DataSplitter(seed=42)
    result = splitter.train_test_split(X, y, val_set=True)
    
    assert len(result) == 6
    X_train, X_val, X_test, y_train, y_val, y_test = result
    assert len(X_train) + len(X_val) + len(X_test) == len(X)

def test_stratified_split(sample_data):
    X, y = sample_data
    splitter = DataSplitter(seed=42)
    X_train, X_test, y_train, y_test = splitter.stratified_train_test_split(X, y)
    
    orig_ratio = y.value_counts(normalize=True)
    train_ratio = y_train.value_counts(normalize=True)
    test_ratio = y_test.value_counts(normalize=True)
    
    assert np.allclose(orig_ratio, train_ratio, atol=0.05)
    assert np.allclose(orig_ratio, test_ratio, atol=0.05)

def test_stratified_split_with_val(sample_data):
    X, y = sample_data
    splitter = DataSplitter(seed=42)
    result = splitter.stratified_train_test_split(X, y, val_set=True)
    
    assert len(result) == 6
    X_train, X_val, X_test, y_train, y_val, y_test = result
    assert len(X_train) + len(X_val) + len(X_test) == len(X)

def test_temporal_split(temporal_data):
    X, y = temporal_data
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test = splitter.temporal_train_test_split(X, y)
    
    # Vérifications
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == 20  # 20% de 100
    
    # Vérifie l'ordre temporel
    last_train_date = X_train.index[-1]
    first_test_date = X_test.index[0]
    assert last_train_date < first_test_date  # Les dates train sont avant test
    
    # Vérifie que les données les plus récentes sont bien dans test
    assert X.index[-1] in X_test.index

def test_temporal_split_with_val(temporal_data):
    X, y = temporal_data
    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.temporal_train_test_split(
        X, y, val_set=True
    )
    
    # Vérifications de taille
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    
    # Vérifie l'ordre temporel: train < val < test
    assert X_train.index[-1] < X_val.index[0]
    assert X_val.index[-1] < X_test.index[0]
    
    # Vérifie que les plus récentes sont dans test
    assert X.index[-1] in X_test.index

def test_k_fold_split(sample_data):
    X, y = sample_data
    splitter = DataSplitter(seed=42)
    folds = splitter.k_fold_split(X, y, k=5)
    
    assert len(folds) == 5
    for X_train, X_test, y_train, y_test in folds:
        assert len(X_train) == 80  # 80% de 100
        assert len(X_test) == 20   # 20% de 100

def test_k_fold_split_no_y(sample_data):
    X, _ = sample_data
    splitter = DataSplitter(seed=42)
    folds = splitter.k_fold_split(X, k=5)
    
    assert len(folds) == 5
    for X_train, X_test in folds:
        assert len(X_train) == 80
        assert len(X_test) == 20

def test_reproducibility(sample_data):
    X, y = sample_data
    splitter1 = DataSplitter(seed=42)
    split1 = splitter1.train_test_split(X, y)
    
    splitter2 = DataSplitter(seed=42)
    split2 = splitter2.train_test_split(X, y)
    
    for arr1, arr2 in zip(split1, split2):
        if isinstance(arr1, pd.DataFrame):
            pd.testing.assert_frame_equal(arr1, arr2)
        else:
            pd.testing.assert_series_equal(arr1, arr2)

def test_stratified_requires_y(sample_data):
    X, _ = sample_data
    splitter = DataSplitter()
    with pytest.raises(AssertionError, match="Stratified split requires target vector y"):
        splitter.stratified_train_test_split(X, None)

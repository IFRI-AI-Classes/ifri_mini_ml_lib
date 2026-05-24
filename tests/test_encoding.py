import pytest
import numpy as np
from ifri_mini_ml_lib.preprocessing.preparation.encoding import OrdinalEncoder
import pandas as pd
from ifri_mini_ml_lib.preprocessing.preparation.encoding import CategoricalEncoder

def test_basic_encoding():
    """Test basic encoding with automatic category detection."""
    data = [['Cat'], ['Dog'], ['Cat'], ['Bird']]
    encoder = OrdinalEncoder(categories='auto')
    
    # Fit & Transform
    encoded = encoder.fit_transform(data)
    
    # Check dimensions and values
    assert encoded.shape == (4, 1)
    # np.unique sorts alphabetically: Bird=0, Cat=1, Dog=2
    assert np.array_equal(encoded, [[1], [2], [1], [0]])

def test_mixed_types():
    """Verify that the encoder handles mixed integers and strings."""
    data = [[1, 'A'], [2, 'B'], [1, 'C']]
    encoder = OrdinalEncoder(categories='auto')
    encoded = encoder.fit_transform(data)
    
    assert encoded.dtype == np.float64
    # Both '1' instances should have the same code
    assert encoded[0, 0] == encoded[2, 0] 

def test_handle_unknown_error():
    """Verify that an error is raised for unknown categories in strict mode."""
    train_data = [['Red'], ['Blue']]
    test_data = [['Green']]
    
    encoder = OrdinalEncoder(handle_unknown='error')
    encoder.fit(train_data)
    
    # Matching the English error message from your OrdinalEncoder class
    with pytest.raises(ValueError, match="Found unknown category 'Green'"):
        encoder.transform(test_data)

def test_handle_unknown_value():
    """Verify the fallback value assignment for unknown categories."""
    train_data = [['Paris'], ['Lyon']]
    test_data = [['Marseille'], ['Paris']] # Marseille is unknown
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(train_data)
    encoded = encoder.transform(test_data)
    
    assert encoded[0, 0] == -1  # Marseille should be mapped to unknown_value
    assert encoded[1, 0] != -1  # Paris should be encoded normally

def test_inverse_transform():
    """Verify that original categories can be recovered."""
    data = [['Small'], ['Large']]
    encoder = OrdinalEncoder(categories='auto')
    encoded = encoder.fit_transform(data)
    
    decoded = encoder.inverse_transform(encoded)
    assert np.array_equal(decoded, data)

def test_manual_categories():
    """Test with manually defined categories (specific order)."""
    # We want 'Cold' to be 0 and 'Hot' to be 1, regardless of alphabetical order
    categories = [['Cold', 'Hot']]
    data = [['Hot'], ['Cold']]
    
    encoder = OrdinalEncoder(categories=categories)
    encoded = encoder.fit_transform(data)
    
    assert encoded[0, 0] == 1 # Hot
    assert encoded[1, 0] == 0 # Cold
    


@pytest.fixture
def sample_data():
    """Fixture providing a simple DataFrame for testing."""
    df = pd.DataFrame({
        'color': ['Red', 'Blue', 'Red', 'Green'],
        'target': [1, 0, 1, 0]
    })
    return df

def test_label_encoding(sample_data):
    """Verifies that Label Encoding assigns unique integers to categories."""
    X = sample_data[['color']]
    encoder = CategoricalEncoder(encoding_type='label')
    
    encoded_df = encoder.fit_transform(X)
    
    # Check that 'Red' consistently gets the same ID
    # Since it uses unique() order, Red=0, Blue=1, Green=2
    assert encoded_df['color'].iloc[0] == encoded_df['color'].iloc[2]
    assert set(encoded_df['color'].unique()) == {0, 1, 2}

def test_frequency_encoding(sample_data):
    """Verifies that categories are replaced by their relative frequency."""
    X = sample_data[['color']]
    encoder = CategoricalEncoder(encoding_type='frequency')
    
    encoded_df = encoder.fit_transform(X)
    
    # Red appears 2/4 times (0.5), Blue 1/4 (0.25), Green 1/4 (0.25)
    assert encoded_df['color'].iloc[0] == 0.5
    assert encoded_df['color'].iloc[1] == 0.25
    assert encoded_df['color'].iloc[3] == 0.25

def test_target_encoding(sample_data):
    """Verifies that categories are replaced by the mean of the target variable."""
    X = sample_data[['color']]
    y = sample_data['target']
    
    # Note: the class uses self.target_column to find the column in the joined DF
    encoder = CategoricalEncoder(encoding_type='target', target_column='target')
    
    encoded_df = encoder.fit_transform(X, y)
    
    # Red: target values are [1, 1] -> mean = 1.0
    # Blue: target value is [0] -> mean = 0.0
    # Green: target value is [0] -> mean = 0.0
    assert encoded_df['color'].iloc[0] == 1.0
    assert encoded_df['color'].iloc[1] == 0.0
    assert encoded_df['color'].iloc[3] == 0.0

def test_target_encoding_missing_y(sample_data):
    """Verifies that a ValueError is raised if y is missing for target encoding."""
    X = sample_data[['color']]
    encoder = CategoricalEncoder(encoding_type='target', target_column='target')
    
    with pytest.raises(ValueError, match="Target encoding requires target column `y`"):
        encoder.fit(X, y=None)

def test_unknown_encoding_type():
    """Verifies that an invalid encoding type raises an error."""
    encoder = CategoricalEncoder(encoding_type='invalid_method')
    df = pd.DataFrame({'col': ['A', 'B']})
    
    with pytest.raises(ValueError, match="Unknown encoding type"):
        encoder.fit(df)
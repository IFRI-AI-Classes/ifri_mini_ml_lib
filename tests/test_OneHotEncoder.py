import pytest
import pandas as pd

from ifri_mini_ml_lib.preprocessing import OneHotEncoder


@pytest.fixture
def sample_dataframe():
    """Provide a sample dataframe for testing."""
    return pd.DataFrame({
        "Color": ["Red", "Blue", "Red"],
        "City": ["Paris", "London", "Paris"],
        "Age": [20, 25, 30]
    })


def test_encoder_initialization():
    """Test encoder initialization with valid and invalid parameters."""

    # Valid initialization
    encoder = OneHotEncoder(handle_unknown="ignore")

    assert encoder.handle_unknown == "ignore"
    assert encoder.categories_ == {}
    assert encoder.feature_names_ == []

    # Invalid initialization
    with pytest.raises(
        ValueError,
        match="handle_unknown must be either 'ignore' or 'error'"
    ):
        OneHotEncoder(handle_unknown="invalid")


def test_fit(sample_dataframe):
    """Test fit method and learned categories."""

    encoder = OneHotEncoder()

    encoder.fit(sample_dataframe)

    # Check learned categories
    assert isinstance(encoder.categories_, dict)

    assert "Color" in encoder.categories_
    assert "City" in encoder.categories_

    assert encoder.categories_["Color"] == ["Blue", "Red"]
    assert encoder.categories_["City"] == ["London", "Paris"]

    # Check generated feature names
    assert "Color_Blue" in encoder.feature_names_
    assert "Color_Red" in encoder.feature_names_
    assert "City_London" in encoder.feature_names_
    assert "City_Paris" in encoder.feature_names_


def test_transform(sample_dataframe):
    """Test transform method output."""

    encoder = OneHotEncoder()

    encoder.fit(sample_dataframe)

    transformed = encoder.transform(sample_dataframe)

    # Check dataframe type
    assert isinstance(transformed, pd.DataFrame)

    # Check generated columns
    assert "Color_Blue" in transformed.columns
    assert "Color_Red" in transformed.columns

    assert "City_London" in transformed.columns
    assert "City_Paris" in transformed.columns

    # Check numerical column is preserved
    assert "Age" in transformed.columns

    # Check encoded values
    assert transformed.loc[0, "Color_Red"] == 1
    assert transformed.loc[0, "Color_Blue"] == 0

    assert transformed.loc[1, "Color_Blue"] == 1
    assert transformed.loc[1, "Color_Red"] == 0


def test_fit_transform(sample_dataframe):
    """Test fit_transform method."""

    encoder = OneHotEncoder()

    transformed = encoder.fit_transform(sample_dataframe)

    assert isinstance(transformed, pd.DataFrame)

    assert "Color_Blue" in transformed.columns
    assert "Color_Red" in transformed.columns


def test_transform_before_fit(sample_dataframe):
    """Test transform before fit raises error."""

    encoder = OneHotEncoder()

    with pytest.raises(
        ValueError,
        match="The encoder has not been fit yet"
    ):
        encoder.transform(sample_dataframe)


def test_unknown_categories_ignore():
    """Test unknown categories with ignore mode."""

    train = pd.DataFrame({
        "Color": ["Red", "Blue"]
    })

    test = pd.DataFrame({
        "Color": ["Green"]
    })

    encoder = OneHotEncoder(handle_unknown="ignore")

    encoder.fit(train)

    transformed = encoder.transform(test)

    # Unknown category should produce all zeros
    assert transformed.loc[0, "Color_Blue"] == 0
    assert transformed.loc[0, "Color_Red"] == 0


def test_unknown_categories_error():
    """Test unknown categories with error mode."""

    train = pd.DataFrame({
        "Color": ["Red", "Blue"]
    })

    test = pd.DataFrame({
        "Color": ["Green"]
    })

    encoder = OneHotEncoder(handle_unknown="error")

    encoder.fit(train)

    with pytest.raises(
        ValueError,
        match="Unknown categories found in column"
    ):
        encoder.transform(test)


def test_empty_dataframe():
    """Test encoder with empty dataframe."""

    df = pd.DataFrame()

    encoder = OneHotEncoder()

    encoder.fit(df)

    transformed = encoder.transform(df)

    assert transformed.empty


def test_missing_values():
    """Test encoder with missing values."""

    df = pd.DataFrame({
        "Color": ["Red", None, "Blue"]
    })

    encoder = OneHotEncoder()

    transformed = encoder.fit_transform(df)

    assert transformed.loc[0, "Color_Red"] == 1
    assert transformed.loc[1, "Color_Red"] == 0
    assert transformed.loc[1, "Color_Blue"] == 0
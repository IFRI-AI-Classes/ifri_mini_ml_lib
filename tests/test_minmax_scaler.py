import pytest
import numpy as np
import pandas as pd
from ifri_mini_ml_lib.preprocessing.preparation import MinMaxScaler  # Assure-toi que le fichier de ta classe est dans le bon dossier

# Exemple de données pour les tests
data = np.array([[1, 2], [2, 4], [3, 6]])
data_df = pd.DataFrame(data, columns=['A', 'B'])

# Test d'initialisation
def test_initialization():
    scaler = MinMaxScaler(feature_range=(0, 1))
    assert scaler.range_min == 0
    assert scaler.range_max == 1

# Test de la méthode fit
def test_fit():
    scaler = MinMaxScaler()
    scaler.fit(data)
    assert np.array_equal(scaler.min_, np.min(data, axis=0))
    assert np.array_equal(scaler.max_, np.max(data, axis=0))

# Test de la méthode transform
def test_transform():
    scaler = MinMaxScaler()
    scaler.fit(data)
    transformed_data = scaler.transform(data)
    assert transformed_data.shape == data.shape
    assert np.allclose(transformed_data, (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)))

# Test de la méthode inverse_transform
def test_inverse_transform():
    scaler = MinMaxScaler()
    scaler.fit(data)
    transformed_data = scaler.transform(data)
    original_data = scaler.inverse_transform(transformed_data)
    assert np.allclose(original_data, data)

# Test de l'erreur si fit n'est pas appelé avant transform
def test_transform_without_fit():
    scaler = MinMaxScaler()
    with pytest.raises(ValueError, match="Le scaler n'a pas été ajusté. Appelez 'fit' avant 'transform'."):
        scaler.transform(data)

# Test de l'erreur si fit n'est pas appelé avant inverse_transform
def test_inverse_transform_without_fit():
    scaler = MinMaxScaler()
    with pytest.raises(ValueError, match="Le scaler n'a pas été ajusté. Appelez 'fit' avant 'inverse_transform'."):
        scaler.inverse_transform(data)

# Test fit_transform en une seule étape
def test_fit_transform():
    scaler = MinMaxScaler()
    transformed_data = scaler.fit_transform(data)
    assert transformed_data.shape == data.shape
    assert np.allclose(transformed_data, (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0)))

# Test de normalisation avec DataFrame
def test_dataframe_input():
    scaler = MinMaxScaler()
    scaler.fit(data_df)
    transformed_data = scaler.transform(data_df)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == data_df.shape

# Test de normalisation avec Series
def test_series_input():
    scaler = MinMaxScaler()
    data_series = pd.Series([1, 2, 3], name='A')
    scaler.fit(data_series)
    transformed_data = scaler.transform(data_series)
    assert isinstance(transformed_data, pd.Series)
    assert transformed_data.shape == data_series.shape

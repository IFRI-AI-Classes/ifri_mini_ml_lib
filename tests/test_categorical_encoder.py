import pytest
import pandas as pd
import numpy as np
from ifri_mini_ml_lib.preprocessing.preparation import CategoricalEncoder

@pytest.fixture
def sample_data():
    """Fournit des données de test pour les encodeurs catégoriels"""
    data = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'blue', 'red'],
        'size': ['S', 'M', 'L', 'M', 'S'],
        'target': [10, 20, 30, 25, 15]
    })
    return data

def test_label_encoding(sample_data):
    """Teste l'encodage par étiquettes"""
    encoder = CategoricalEncoder(encoding_type='label')
    X = sample_data[['color', 'size']]
    encoded = encoder.fit_transform(X)
    
   
    assert encoded['color'].dtype in [np.int64, np.int32]
    assert encoded['size'].dtype in [np.int64, np.int32]
    
    
    assert set(encoded['color'].unique()) == {0, 1, 2}
    assert set(encoded['size'].unique()) == {0, 1, 2}

def test_ordinal_encoding(sample_data):
    """Teste l'encodage ordinal"""
    encoder = CategoricalEncoder(encoding_type='ordinal')
    X = sample_data[['color', 'size']]
    encoded = encoder.fit_transform(X)
    
    
    assert encoder.mapping['color']['blue'] == 0  
    assert encoder.mapping['size']['L'] == 2      

def test_frequency_encoding(sample_data):
    """Teste l'encodage par fréquence"""
    encoder = CategoricalEncoder(encoding_type='frequency')
    X = sample_data[['color', 'size']]
    encoded = encoder.fit_transform(X)
    
    # Calcul des fréquences attendues
    color_freqs = sample_data['color'].value_counts(normalize=True)
    size_freqs = sample_data['size'].value_counts(normalize=True)
    
    # Vérifie que les valeurs sont correctement remplacées
    for idx, row in sample_data.iterrows():
        assert encoded.loc[idx, 'color'] == color_freqs[row['color']]
        assert encoded.loc[idx, 'size'] == size_freqs[row['size']]
    
    # Vérifie la moyenne calculée (0.4*2 + 0.4*2 + 0.2*1)/5 = (0.8 + 0.8 + 0.2)/5 = 1.8/5 = 0.36
    expected_color_mean = (2*color_freqs['red'] + 2*color_freqs['blue'] + color_freqs['green'])/5
    assert pytest.approx(encoded['color'].mean()) == expected_color_mean
    
    
def test_target_encoding(sample_data):
    """Teste l'encodage par cible"""
    encoder = CategoricalEncoder(encoding_type='target', target_column='target')
    X = sample_data[['color', 'size']]
    y = sample_data['target']
    encoded = encoder.fit_transform(X, y)
    
    # Vérifie que les valeurs sont remplacées par la moyenne de la cible
    # Pour 'color': red=(10+15)/2=12.5, blue=(20+25)/2=22.5, green=30
    assert pytest.approx(encoded.loc[0, 'color']) == 12.5
    assert pytest.approx(encoded.loc[1, 'color']) == 22.5
    assert pytest.approx(encoded.loc[2, 'color']) == 30.0

def test_onehot_encoding(sample_data):
    """Teste l'encodage one-hot"""
    encoder = CategoricalEncoder(encoding_type='onehot')
    X = sample_data[['color', 'size']]
    encoded = encoder.fit_transform(X)
    
    # Vérifie que les colonnes originales sont supprimées
    assert 'color' not in encoded.columns
    assert 'size' not in encoded.columns
    
    # Vérifie que les nouvelles colonnes one-hot sont créées
    expected_columns = {'color_red', 'color_blue', 'color_green', 'size_S', 'size_M', 'size_L'}
    assert expected_columns.issubset(set(encoded.columns))
    
    # Vérifie les valeurs one-hot
    assert encoded['color_red'].sum() == 2
    assert encoded['size_L'].sum() == 1

def test_missing_target_for_target_encoding(sample_data):
    """Teste que l'encodage par cible lève une erreur si pas de cible fournie"""
    encoder = CategoricalEncoder(encoding_type='target', target_column='target')
    X = sample_data[['color', 'size']]
    
    with pytest.raises(ValueError, match="Target encoding requires target column `y`"):
        encoder.fit(X)

def test_unknown_encoding_type(sample_data):
    """Teste qu'une erreur est levée pour un type d'encodage inconnu"""
    with pytest.raises(ValueError, match="Unknown encoding type: unknown"):
        encoder = CategoricalEncoder(encoding_type='unknown')
        encoder.fit(sample_data[['color', 'size']])

def test_transform_before_fit(sample_data):
    """Teste qu'une erreur est levée si on transforme avant d'avoir fit"""
    encoder = CategoricalEncoder(encoding_type='label')
    with pytest.raises(KeyError):
        encoder.transform(sample_data[['color', 'size']])

def test_handle_unseen_categories(sample_data):
    """Teste le comportement avec des catégories non vues pendant le fit"""
    encoder = CategoricalEncoder(encoding_type='label')
    X_train = sample_data[['color', 'size']]
    encoder.fit(X_train)
    
    # Données de test avec une nouvelle catégorie
    X_test = pd.DataFrame({
        'color': ['red', 'yellow'],  # 'yellow' pas vu pendant le fit
        'size': ['S', 'M']
    })
    
    encoded = encoder.transform(X_test)
    
    # Vérifie que la nouvelle catégorie est encodée comme NaN
    assert pd.isna(encoded.loc[1, 'color'])
    assert encoded.loc[0, 'size'] == 0  # 'S' devrait être encodé comme 0
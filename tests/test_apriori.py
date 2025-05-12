import pytest
from ifri_mini_ml_lib.association_rules import Apriori

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

def test_apriori_initialization():
    """Test Apriori initialization with valid and invalid parameters."""
    # Valid initialization
    ap = Apriori(min_support=0.4, min_confidence=0.6)
    assert ap.min_support == 0.4
    assert ap.min_confidence == 0.6

    # Invalid min_support
    with pytest.raises(ValueError, match="Minimum support must be between 0 and 1"):
        Apriori(min_support=1.5, min_confidence=0.6)
    
    # Invalid min_confidence
    with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1"):
        Apriori(min_support=0.4, min_confidence=-0.1)

def test_apriori_fit(transactions):
    """Test Apriori fit method and frequent itemsets generation."""
    ap = Apriori(min_support=0.4, min_confidence=0.6)
    ap.fit(transactions)
    
    # Check if frequent itemsets are generated
    frequent_itemsets = ap.get_frequent_itemsets()
    assert isinstance(frequent_itemsets, dict)
    assert len(frequent_itemsets) >= 1  # At least size-1 itemsets
    assert 1 in frequent_itemsets
    assert len(frequent_itemsets[1]) >= 3  # Expect at least bread, milk, butter
    
    # Verify specific frequent itemsets from example
    size_1_itemsets = [list(item)[0] for item in frequent_itemsets[1]]
    assert 'bread' in size_1_itemsets
    assert 'milk' in size_1_itemsets
    assert 'butter' in size_1_itemsets

def test_apriori_rules(transactions):
    """Test Apriori association rules generation."""
    ap = Apriori(min_support=0.4, min_confidence=0.6)
    ap.fit(transactions)
    
    rules = ap.get_rules()
    assert isinstance(rules, list)
    assert len(rules) >= 2  # Expect at least the rules from example
    for rule in rules:
        assert 'antecedent' in rule
        assert 'consequent' in rule
        assert 'support' in rule
        assert 'confidence' in rule
        assert 'lift' in rule
        assert rule['confidence'] >= 0.6  # Matches min_confidence
        assert rule['support'] >= 0.4  # Matches min_support
        assert rule['lift'] >= 0
    
    # Verify specific rules from example
    rule_sets = [(set(rule['antecedent']), set(rule['consequent'])) for rule in rules]
    assert ({'milk'}, {'bread'}) in rule_sets
    assert ({'bread'}, {'milk'}) in rule_sets
    for rule in rules:
        if set(rule['antecedent']) == {'milk'} and set(rule['consequent']) == {'bread'}:
            assert abs(rule['support'] - 0.6) < 1e-6
            assert abs(rule['confidence'] - 0.75) < 1e-6

def test_apriori_invalid_input():
    """Test Apriori with invalid input data."""
    ap = Apriori(min_support=0.4, min_confidence=0.6)
    with pytest.raises(TypeError, match="Data format not respected! Only List\\[set\\] format is accepted."):
        ap.fit("invalid_data")

def test_apriori_empty_transactions():
    """Test Apriori with empty transactions."""
    ap = Apriori(min_support=0.4, min_confidence=0.6)
    ap.fit([])
    assert ap.get_frequent_itemsets() == {}
    assert ap.get_rules() == []
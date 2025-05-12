import pytest
from ifri_mini_ml_lib.association_rules import Eclat

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

def test_eclat_initialization():
    """Test Eclat initialization with valid and invalid parameters."""
    # Valid initialization
    ec = Eclat(min_support=0.4, min_confidence=0.6)
    assert ec.min_support == 0.4
    assert ec.min_confidence == 0.6

    # Invalid min_support
    with pytest.raises(ValueError, match="Minimum support must be between 0 and 1"):
        Eclat(min_support=1.5, min_confidence=0.6)
    
    # Invalid min_confidence
    with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1"):
        Eclat(min_support=0.4, min_confidence=-0.1)

def test_eclat_fit(transactions):
    """Test Eclat fit method and frequent itemsets generation."""
    ec = Eclat(min_support=0.4, min_confidence=0.6)
    ec.fit(transactions)
    
    # Check if frequent itemsets are generated
    frequent_itemsets = ec.get_frequent_itemsets()
    assert isinstance(frequent_itemsets, dict)
    assert len(frequent_itemsets) >= 1  # At least size-1 itemsets
    assert 1 in frequent_itemsets
    assert len(frequent_itemsets[1]) >= 3  # Expect at least bread, milk, butter
    
    # Verify specific frequent itemsets from example
    for itemset, tidset in frequent_itemsets[1].items():
        item = list(itemset)[0]
        support = len(tidset) / ec.n_transactions
        if item == 'bread':
            assert abs(support - 0.8) < 1e-6
        elif item == 'milk':
            assert abs(support - 0.8) < 1e-6
        elif item == 'butter':
            assert abs(support - 0.6) < 1e-6

def test_eclat_rules(transactions):
    """Test Eclat association rules generation."""
    ec = Eclat(min_support=0.4, min_confidence=0.6)
    ec.fit(transactions)
    
    rules = ec.get_rules()
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
            assert abs(rule['confidence'] - 0.75) < 1e-6
            assert abs(rule['lift'] - 0.9375) < 1e-6

def test_eclat_invalid_input():
    """Test Eclat with invalid input data."""
    ec = Eclat(min_support=0.4, min_confidence=0.6)
    with pytest.raises(TypeError, match="Data format not respected! Only List\\[set\\] format is accepted."):
        ec.fit("invalid_data")

def test_eclat_empty_transactions():
    """Test Eclat with empty transactions."""
    ec = Eclat(min_support=0.4, min_confidence=0.6)
    ec.fit([])
    assert ec.get_frequent_itemsets() == {}
    assert ec.get_rules() == []
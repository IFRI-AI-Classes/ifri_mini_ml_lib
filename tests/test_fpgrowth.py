import pytest
from ifri_mini_ml_lib.association_rules import FPGrowth

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

def test_fp_growth_initialization():
    """Test FP-Growth initialization with valid and invalid parameters."""
    # Valid initialization
    fp = FPGrowth(min_support=0.4, min_confidence=0.6)
    assert fp.min_support == 0.4
    assert fp.min_confidence == 0.6

    # Invalid min_support
    with pytest.raises(ValueError, match="Minimum support must be between 0 and 1"):
        FPGrowth(min_support=1.5, min_confidence=0.6)
    
    # Invalid min_confidence
    with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1"):
        FPGrowth(min_support=0.4, min_confidence=-0.1)

def test_fp_growth_fit(transactions):
    """Test FP-Growth fit method and frequent itemsets generation."""
    fp = FPGrowth(min_support=0.4, min_confidence=0.6)
    fp.fit(transactions)
    
    # Check if frequent itemsets are generated
    frequent_itemsets = fp.get_frequent_itemsets()
    assert len(frequent_itemsets) >= 3  # Expect at least bread, milk, butter
    assert all('itemset' in item and 'support' in item for item in frequent_itemsets)
    
    # Verify specific frequent itemsets from example
    itemsets = [set(item['itemset']) for item in frequent_itemsets]
    supports = {frozenset(item['itemset']): item['support'] for item in frequent_itemsets}
    assert {'bread'} in itemsets
    assert {'milk'} in itemsets
    assert {'butter'} in itemsets
    assert abs(supports[frozenset({'bread'})] - 0.8) < 1e-6
    assert abs(supports[frozenset({'milk'})] - 0.8) < 1e-6
    assert abs(supports[frozenset({'butter'})] - 0.6) < 1e-6

def test_fp_growth_rules(transactions):
    """Test FP-Growth association rules generation."""
    fp = FPGrowth(min_support=0.4, min_confidence=0.6)
    fp.fit(transactions)
    
    rules = fp.get_rules()
    assert isinstance(rules, list)
    assert len(rules) >= 2  # Expect at least the rules from example
    for rule in rules:
        assert 'antecedent' in rule
        assert 'consequent' in rule
        assert 'confidence' in rule
        assert 'lift' in rule
        assert rule['confidence'] >= 0.6  # Matches min_confidence
        assert rule['lift'] >= 0
    
    # Verify specific rules from example
    rule_sets = [(set(rule['antecedent']), set(rule['consequent'])) for rule in rules]
    assert ({'milk'}, {'bread'}) in rule_sets
    assert ({'bread'}, {'milk'}) in rule_sets
    for rule in rules:
        if set(rule['antecedent']) == {'milk'} and set(rule['consequent']) == {'bread'}:
            assert abs(rule['confidence'] - 0.75) < 1e-6
            assert abs(rule['lift'] - 0.9375) < 1e-6

def test_fp_growth_invalid_input():
    """Test FP-Growth with invalid input data."""
    fp = FPGrowth(min_support=0.4, min_confidence=0.6)
    with pytest.raises(TypeError, match="Data format not respected! Only List\\[set\\] format is accepted."):
        fp.fit("invalid_data")

def test_fp_growth_empty_transactions():
    """Test FP-Growth with empty transactions."""
    fp = FPGrowth(min_support=0.4, min_confidence=0.6)
    fp.fit([])
    assert len(fp.get_frequent_itemsets()) == 0
    assert len(fp.get_rules()) == 0
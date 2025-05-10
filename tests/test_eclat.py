import pytest
from ifri_mini_ml_lib.association_rules import Eclat, DataAdapter

@pytest.fixture
def transactions():
    """Load transactions from Market_Basket_Optimisation.csv."""
    transactions = DataAdapter.load_csv_to_transactions("tests/data/Market_Basket_Optimisation.csv")
    return transactions

def test_eclat_initialization():
    """Test Eclat initialization with valid and invalid parameters."""
    # Valid initialization
    ec = Eclat(min_support=0.05, min_confidence=0.5)
    assert ec.min_support == 0.05
    assert ec.min_confidence == 0.5

    # Invalid min_support
    with pytest.raises(ValueError, match="Minimum support must be between 0 and 1"):
        Eclat(min_support=1.5, min_confidence=0.5)
    
    # Invalid min_confidence
    with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1"):
        Eclat(min_support=0.05, min_confidence=-0.1)

def test_eclat_fit(transactions):
    """Test Eclat fit method and frequent itemsets generation."""
    ec = Eclat(min_support=0.05, min_confidence=0.5)
    ec.fit(transactions)
    
    # Check if frequent itemsets are generated
    frequent_itemsets = ec.get_frequent_itemsets()
    assert isinstance(frequent_itemsets, dict)
    assert len(frequent_itemsets) > 0
    for size, itemsets in frequent_itemsets.items():
        assert isinstance(itemsets, dict)
        for itemset, tidset in itemsets.items():
            assert isinstance(itemset, frozenset)
            assert isinstance(tidset, set)
            assert len(tidset) / ec.n_transactions >= 0.05  # Matches min_support

def test_eclat_rules(transactions):
    """Test Eclat association rules generation."""
    ec = Eclat(min_support=0.05, min_confidence=0.5)
    ec.fit(transactions)
    
    rules = ec.get_rules()
    assert isinstance(rules, list)
    if rules:
        for rule in rules:
            assert 'antecedent' in rule
            assert 'consequent' in rule
            assert 'support' in rule
            assert 'confidence' in rule
            assert 'lift' in rule
            assert rule['confidence'] >= 0.5  # Matches min_confidence
            assert rule['support'] >= 0.05  # Matches min_support
            assert rule['lift'] >= 0

def test_eclat_invalid_input():
    """Test Eclat with invalid input data."""
    ec = Eclat(min_support=0.05, min_confidence=0.5)
    with pytest.raises(TypeError, match="Data format not respected! Only List\\[set\\] format is accepted."):
        ec.fit("invalid_data")

def test_eclat_empty_transactions():
    # Test Eclat with empty transactions.
    ec = Eclat(min_support=0.05, min_confidence=0.5)
    ec.fit([])
    assert ec.get_frequent_itemsets() == {}
    assert ec.get_rules() == []

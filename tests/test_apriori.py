import pytest
from ifri_mini_ml_lib.association_rules import Apriori, DataAdapter

@pytest.fixture
def transactions():
    """Load transactions from Market_Basket_Optimisation.csv."""
    transactions = DataAdapter.load_csv_to_transactions(file_path="tests/data/Market_Basket_Optimisation.csv")
    return transactions

def test_apriori_initialization():
    """Test Apriori initialization with valid and invalid parameters."""
    # Valid initialization
    ap = Apriori(min_support=0.05, min_confidence=0.5)
    assert ap.min_support == 0.05
    assert ap.min_confidence == 0.5

    # Invalid min_support
    with pytest.raises(ValueError, match="Minimum support must be between 0 and 1"):
        Apriori(min_support=1.5, min_confidence=0.5)
    
    # Invalid min_confidence
    with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1"):
        Apriori(min_support=0.05, min_confidence=-0.1)

def test_apriori_fit(transactions):
    """Test Apriori fit method and frequent itemsets generation."""
    ap = Apriori(min_support=0.05, min_confidence=0.5)
    ap.fit(transactions)
    
    # Check if frequent itemsets are generated
    frequent_itemsets = ap.get_frequent_itemsets()
    assert isinstance(frequent_itemsets, dict)
    assert len(frequent_itemsets) > 0
    for size, itemsets in frequent_itemsets.items():
        assert isinstance(itemsets, set)
        assert all(isinstance(itemset, frozenset) for itemset in itemsets)

def test_apriori_rules(transactions):
    """Test Apriori association rules generation."""
    ap = Apriori(min_support=0.05, min_confidence=0.5)
    ap.fit(transactions)
    
    rules = ap.get_rules()
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

def test_apriori_invalid_input():
    """Test Apriori with invalid input data."""
    ap = Apriori(min_support=0.05, min_confidence=0.5)
    with pytest.raises(TypeError, match="Data format not respected! Only List\\[set\\] format is accepted."):
        ap.fit("invalid_data")

def test_apriori_empty_transactions():
    # Test Apriori with empty transactions.
    ap = Apriori(min_support=0.05, min_confidence=0.5)
    ap.fit([])
    assert ap.get_frequent_itemsets() == {}
    assert ap.get_rules() == []

import pytest
from ifri_mini_ml_lib.association_rules import FPGrowth, DataAdapter

@pytest.fixture
def transactions():
    """Load transactions from Market_Basket_Optimisation.csv."""
    transactions = DataAdapter.load_csv_to_transactions("tests/data/Market_Basket_Optimisation.csv")
    return transactions

def test_fp_growth_initialization():
    """Test FP-Growth initialization with valid and invalid parameters."""
    # Valid initialization
    fp = FPGrowth(min_support=0.05, min_confidence=0.5)
    assert fp.min_support == 0.05
    assert fp.min_confidence == 0.5

    # Invalid min_support
    with pytest.raises(ValueError, match="Minimum support must be between 0 and 1"):
        FPGrowth(min_support=1.5, min_confidence=0.5)
    
    # Invalid min_confidence
    with pytest.raises(ValueError, match="Minimum confidence must be between 0 and 1"):
        FPGrowth(min_support=0.05, min_confidence=-0.1)

def test_fp_growth_fit(transactions):
    """Test FP-Growth fit method and frequent itemsets generation."""
    fp = FPGrowth(min_support=0.05, min_confidence=0.5)
    fp.fit(transactions)
    
    # Check if frequent itemsets are generated
    frequent_itemsets = fp.get_frequent_itemsets()
    assert len(frequent_itemsets) > 0
    assert all('itemset' in item and 'support' in item for item in frequent_itemsets)
    
    # Verify support values are within valid range
    for itemset_data in frequent_itemsets:
        assert 0 <= itemset_data['support'] <= 1
        assert itemset_data['support'] >= 0.05  # Matches min_support

def test_fp_growth_rules(transactions):
    """Test FP-Growth association rules generation."""
    fp = FPGrowth(min_support=0.05, min_confidence=0.5)
    fp.fit(transactions)
    
    rules = fp.get_rules()
    assert isinstance(rules, list)
    if rules:
        for rule in rules:
            assert 'antecedent' in rule
            assert 'consequent' in rule
            assert 'confidence' in rule
            assert 'lift' in rule
            assert rule['confidence'] >= 0.5  # Matches min_confidence
            assert rule['lift'] >= 0

def test_fp_growth_invalid_input():
    """Test FP-Growth with invalid input data."""
    fp = FPGrowth(min_support=0.05, min_confidence=0.5)
    with pytest.raises(TypeError, match="Data format not respected! Only List\\[set\\] format is accepted."):
        fp.fit("invalid_data")

def test_fp_growth_empty_transactions():
    """Test FP-Growth with empty transactions."""
    fp = FPGrowth(min_support=0.05, min_confidence=0.5)
    fp.fit([])
    assert len(fp.get_frequent_itemsets()) == 0
    assert len(fp.get_rules()) == 0
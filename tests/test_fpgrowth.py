import pytest
from ifri_mini_ml_lib.association_rules import FPGrowth

def test_fpgrowth_basic():
    transactions = [
        {'bread', 'milk', 'butter'},
        {'bread', 'jam', 'eggs'},
        {'milk', 'butter', 'cheese'},
        {'bread', 'milk', 'butter', 'cheese'},
        {'bread', 'jam', 'milk'}
    ]
    model = FPGrowth(min_support=0.4, min_confidence=0.6)
    model.fit(transactions)
    frequent_itemsets = model.get_frequent_itemsets()
    rules = model.get_rules()

    # Check that at least some frequent itemsets are found
    assert isinstance(frequent_itemsets, list)
    assert any('bread' in itemset['itemset'] for itemset in frequent_itemsets)
    assert any('milk' in itemset['itemset'] for itemset in frequent_itemsets)

    # Check that rules are generated and have expected keys
    if rules:
        for rule in rules:
            assert 'antecedent' in rule
            assert 'consequent' in rule
            assert 'support' in rule
            assert 'confidence' in rule
            assert 'lift' in rule

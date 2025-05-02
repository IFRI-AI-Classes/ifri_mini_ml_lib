import pytest
from ifri_mini_ml_lib.association_rules import Apriori

def test_apriori_basic():
    transactions = [
        {'bread', 'milk', 'butter'},
        {'bread', 'jam', 'eggs'},
        {'milk', 'butter', 'cheese'},
        {'bread', 'milk', 'butter', 'cheese'},
        {'bread', 'jam', 'milk'}
    ]
    model = Apriori(min_support=0.4, min_confiance=0.6)
    model.fit(transactions)
    frequent_itemsets = model.get_frequent_itemsets()
    rules = model.get_rules()

    # Check frequent itemsets of size 1
    assert 1 in frequent_itemsets
    items = set()
    for itemset in frequent_itemsets[1]:
        items.update(itemset)
    assert 'bread' in items
    assert 'milk' in items
    assert 'butter' in items

    # Check that rules are generated and have expected keys
    if rules:
        for rule in rules:
            assert 'antecedent' in rule
            assert 'consequent' in rule
            assert 'support' in rule
            assert 'confidence' in rule
            assert 'lift' in rule

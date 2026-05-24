import pytest
import pandas as pd

from ifri_mini_ml_lib.association_rules.association_rules import AssociationRules
from ifri_mini_ml_lib.metrics.rules import evaluate_rule_from_supports


def test_evaluate_rule_from_supports():
    metrics = evaluate_rule_from_supports(0.8, 0.8, 0.6)

    assert metrics["support"] == pytest.approx(0.6)
    assert metrics["confidence"] == pytest.approx(0.75)
    assert metrics["lift"] == pytest.approx(0.9375)
    assert metrics["conviction"] == pytest.approx(0.8)


def test_association_rules_generate_rules_uses_support_metrics():
    frequent_itemsets = pd.DataFrame(
        [
            {"itemsets": frozenset({"bread"}), "support": 0.8},
            {"itemsets": frozenset({"milk"}), "support": 0.8},
            {"itemsets": frozenset({"bread", "milk"}), "support": 0.6},
        ]
    )

    rules = AssociationRules(min_confidence=0.5, min_lift=0.9).generate_rules(frequent_itemsets)

    assert list(rules["antecedents"]) == [frozenset({"bread"}), frozenset({"milk"})]
    assert list(rules["consequents"]) == [frozenset({"milk"}), frozenset({"bread"})]
    assert rules.loc[0, "confidence"] == pytest.approx(0.75)
    assert rules.loc[0, "lift"] == pytest.approx(0.9375)
    assert rules.loc[0, "conviction"] == pytest.approx(0.8)
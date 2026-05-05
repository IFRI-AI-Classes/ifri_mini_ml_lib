"""
Rule generation from frequent itemsets.

This module is algorithm-agnostic: it accepts the output (a DataFrame of
frequent itemsets with their support) produced by *any* mining algorithm
(Apriori, ECLAT, FP-Growth) and derives association rules filtered by
confidence, lift, and optionally conviction.

Metrics:

- Support   : P(X ∪ Y)
- Confidence: P(Y | X) = Support(X ∪ Y) / Support(X)
- Lift      : Confidence(X→Y) / Support(Y)  
- Conviction: (1 − Support(Y)) / (1 − Confidence(X→Y))
"""

from itertools import combinations
import time
import pandas as pd

from ..metrics.rules import evaluate_rule_from_supports


class AssociationRules:
    """
    Generate association rules from a collection of frequent itemsets.

    The class is intentionally decoupled from any specific mining algorithm.
    It consumes the DataFrame produced by ``Apriori``, ``ECLAT``, or
    ``FPGrowth`` and enumerates all valid antecedent/consequent splits.

    Args:
    
    min_confidence : float, default=0.5
        Minimum confidence threshold in [0, 1].
    min_lift : float, default=1.0
        Minimum lift threshold.  Values above 1.0 indicate positive
        correlation beyond what is expected by chance.
    min_conviction : float or None, default=None
        Optional minimum conviction threshold.  When None, conviction
        is computed but not used as a filter.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
        min_conviction: float | None = None,
    ):
        """
        Initialize the AssociationRules generator.
        
        Args:
            min_confidence (float): Minimum confidence threshold in (0, 1]. Defaults to 0.5.
            min_lift (float): Minimum lift threshold (non-negative). Defaults to 1.0.
            min_conviction (float | None): Optional minimum conviction threshold. Defaults to None.
        
        Raises:
            ValueError: If min_confidence not in (0, 1] or min_lift is negative.
        """
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1].")
        if min_lift < 0:
            raise ValueError("min_lift must be non-negative.")

        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.min_conviction = min_conviction

        self._rules: pd.DataFrame | None = None
        self._execution_time: float = 0.0

    def generate_rules(self, frequent_itemsets: pd.DataFrame) -> pd.DataFrame:
        """
        Generate association rules from a frequent-itemsets DataFrame.

        Args:
            frequent_itemsets (pd.DataFrame): DataFrame with columns 'itemsets' (frozenset)
                and 'support' (float). Typically from Apriori, ECLAT, or FPGrowth.

        Returns:
            pd.DataFrame: DataFrame with columns 'antecedents', 'consequents', 
                'antecedent_support', 'consequent_support', 'support', 'confidence', 
                'lift', 'conviction'. Sorted by confidence and lift (descending).
        
        Raises:
            ValueError: If input DataFrame is malformed or empty.
        """
        start_time = time.time()
        self._validate_input(frequent_itemsets)

        # Build a fast lookup: frozenset -> support
        support_lookup: dict[frozenset, float] = {
            row["itemsets"]: row["support"]
            for _, row in frequent_itemsets.iterrows()
        }

        records = []
        # Only itemsets of size ≥ 2 can produce rules
        multi_item = frequent_itemsets[frequent_itemsets["itemsets"].apply(len) >= 2]

        for _, row in multi_item.iterrows():
            itemset: frozenset = row["itemsets"]
            itemset_support: float = row["support"]

            # Enumerate all non-empty proper subsets as antecedents
            for size in range(1, len(itemset)):
                for antecedent_tuple in combinations(sorted(itemset), size):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent

                    ant_support = support_lookup.get(antecedent)
                    cons_support = support_lookup.get(consequent)

                    # Both sub-itemsets must have been frequent
                    if ant_support is None or cons_support is None:
                        continue

                    metrics = evaluate_rule_from_supports(
                        antecedent_support=ant_support,
                        consequent_support=cons_support,
                        rule_support=itemset_support,
                    )
                    confidence = metrics["confidence"]
                    if confidence < self.min_confidence:
                        continue

                    lift = metrics["lift"]
                    if lift < self.min_lift:
                        continue

                    conviction = metrics["conviction"]

                    if (
                        self.min_conviction is not None
                        and conviction is not None
                        and conviction < self.min_conviction
                    ):
                        continue

                    records.append(
                        {
                            "antecedents": antecedent,
                            "consequents": consequent,
                            "antecedent_support": ant_support,
                            "consequent_support": cons_support,
                            "support": itemset_support,
                            "confidence": confidence,
                            "lift": lift,
                            "conviction": conviction,
                        }
                    )

        if not records:
            self._rules = pd.DataFrame(
                columns=[
                    "antecedents", "consequents", "antecedent_support",
                    "consequent_support", "support", "confidence", "lift",
                    "conviction",
                ]
            )
        else:
            self._rules = (
                pd.DataFrame(records)
                .sort_values("lift", ascending=False)
                .reset_index(drop=True)
            )

        self._execution_time = time.time() - start_time
        return self._rules

    # Alias for scikit-learn-style pipeline compatibility
    fit = generate_rules

    def get_rules(self) -> pd.DataFrame:
        """Return the last set of generated rules."""
        if self._rules is None:
            raise RuntimeError("Call generate_rules() first.")
        return self._rules
    
    @property
    def execution_time(self) -> float:
        """Execution time (in seconds) of the last generate_rules() call."""
        return self._execution_time

    def summary(self) -> None:
        """Print a concise summary of the generated rules."""
        if self._rules is None:
            raise RuntimeError("Call generate_rules() first.")

        df = self._rules
        print(f"Total rules generated : {len(df)}")
        if len(df) == 0:
            return
        print(f"Confidence  — min: {df['confidence'].min():.4f}  "
            f"max: {df['confidence'].max():.4f}  "
            f"mean: {df['confidence'].mean():.4f}")
        print(f"Lift        — min: {df['lift'].min():.4f}  "
            f"max: {df['lift'].max():.4f}  "
            f"mean: {df['lift'].mean():.4f}")
        if df["conviction"].notna().any():
            finite = df["conviction"].replace(float("inf"), pd.NA).dropna()
            if len(finite):
                print(f"Conviction  — min: {finite.min():.4f}  "
                    f"max: {finite.max():.4f}  "
                    f"mean: {finite.mean():.4f}  "
                    f"(∞ rules: {(df['conviction'] == float('inf')).sum()})")


    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """Raise informative errors for malformed input DataFrames."""
        required = {"itemsets", "support"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"frequent_itemsets DataFrame is missing columns: {missing}. "
                "Make sure to pass the output of get_frequent_itemsets()."
            )
        if df.empty:
            raise ValueError("frequent_itemsets DataFrame is empty.")
        if not df["itemsets"].apply(lambda x: isinstance(x, frozenset)).all():
            raise TypeError(
                "Column 'itemsets' must contain frozenset objects. "
                "Use get_frequent_itemsets() from Apriori / ECLAT / FPGrowth."
            )

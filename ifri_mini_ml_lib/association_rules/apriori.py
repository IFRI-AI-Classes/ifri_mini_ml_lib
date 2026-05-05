from itertools import combinations
from collections import defaultdict
import time
import pandas as pd


class Apriori:
    """
    The Apriori algorithm is used to discover frequent itemsets
    in large transactional datasets. For more details: 
    `Agrawal, R., & Srikant, R. (1994, September) <http://www.vldb.org/conf/1994/P487.PDF>`_

    The algorithm performs a breadth-first search over itemsets of increasing
    size.  At each level k it:
    1- Generates candidate k-itemsets from the frequent (k-1)-itemsets
        (join step).
    2- Prunes any candidate whose (k-1)-subsets are not all frequent
        (prune step, based on the anti-monotonicity of support).
    3- Counts candidate support by scanning the transaction database.
    4- Retains only candidates whose support meets *min_support*.
    
    Args:
        min_support (float, optional): Minimum support threshold in [0, 1]. Defaults to 0.5.
        min_confidence (float | None, optional): Optional confidence threshold in (0, 1]
            kept for API consistency with rule-generation workflows. Defaults to None.
        
    Example:

    >>> transactions = [
    ...     {'bread', 'milk', 'butter'},
    ...     {'bread', 'jam', 'eggs'},
    ...     {'milk', 'butter', 'cheese'},
    ...     {'bread', 'milk', 'butter', 'cheese'},
    ...     {'bread', 'jam', 'milk'}
    ... ]
    >>> from ifri_mini_ml_lib.association_rules import Apriori
    >>> apriori = Apriori(min_support=0.4, min_confidence=0.6)
    >>> apriori.fit(transactions)
    <ifri_mini_ml_lib.association_rules.apriori.Apriori object>
    >>> frequent_itemsets = apriori.get_frequent_itemsets()
    >>> print(frequent_itemsets.head(3))
    """

    def __init__(self, min_support: float = 0.5, min_confidence: float | None = None):
        """
        Initialize the Apriori algorithm.
        
        Args:
            min_support (float): Minimum support threshold in (0, 1]. Defaults to 0.5.
            min_confidence (float | None): Optional confidence threshold in (0, 1].
                Defaults to None.
        
        Raises:
            ValueError: If min_support is not in (0, 1] or min_confidence is invalid.
        """
        if not 0.0 < min_support <= 1.0:
            raise ValueError("min_support must be between 0 and 1.")
        if min_confidence is not None and not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1].")
        
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._frequent_itemsets: dict[frozenset, float] = {}  
        self._n_transactions: int = 0
        self._execution_time: float = 0.0

    def fit(self, transactions: list[list]) -> "Apriori":
        """
        Mine frequent itemsets from a list of transactions.

        Args:
            transactions (list[list]): Each inner list is one transaction containing hashable items
                (strings, ints, etc.).

        Returns:
            Apriori: The current instance for method chaining.
        
        Raises:
            ValueError: If transactions list is empty.
        """
        start_time = time.time()
        if not transactions:
            raise ValueError("transactions must not be empty.")

        # Convert once to frozensets for fast subset checks
        encoded = [frozenset(t) for t in transactions]
        self._n_transactions = len(encoded)
        self._frequent_itemsets = {}

        # For single items 
        freq_k = self._get_frequent_1_itemsets(encoded)
        self._frequent_itemsets.update(freq_k)

        # for k >= 2
        k = 2
        while freq_k:
            candidates = self._generate_candidates(freq_k, k)
            if not candidates:
                break
            freq_k = self._count_and_filter(encoded, candidates)
            self._frequent_itemsets.update(freq_k)
            k += 1

        self._execution_time = time.time() - start_time
        return self

    def get_frequent_itemsets(self) -> pd.DataFrame:
        """
        Return the mined frequent itemsets as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with columns 'itemsets' (frozenset), 'support' (float), 
                and 'length' (int). Sorted by itemset length then support.
        
        Raises:
            RuntimeError: If fit() has not been called yet.
        """
        if not self._frequent_itemsets:
            raise RuntimeError("Call fit() before get_frequent_itemsets().")

        records = [
            {"itemsets": itemset, "support": support}
            for itemset, support in self._frequent_itemsets.items()
        ]
        df = pd.DataFrame(records)
        df["length"] = df["itemsets"].apply(len)
        return df.sort_values(["length", "support"], ascending=[True, False]).reset_index(drop=True)

    @property
    def n_transactions(self) -> int:
        """Number of transactions seen during fit()."""
        return self._n_transactions
    
    @property
    def execution_time(self) -> float:
        """Execution time (in seconds) of the last fit() call."""
        return self._execution_time

    def _get_frequent_1_itemsets(
        self, encoded: list[frozenset]
    ) -> dict[frozenset, float]:
        
        """
        Count single-item support and apply min_support filter.
        
        Args:
            encoded: List of transactions, where each transaction is a frozenset of items.
        Returns:
            Dictionary mapping frequent 1-itemsets (frozensets) to their support.
        
        """
        counts: dict[frozenset, int] = defaultdict(int)
        for transaction in encoded:
            for item in transaction:
                counts[frozenset([item])] += 1

        n = self._n_transactions
        return {
            itemset: count / n
            for itemset, count in counts.items()
            if count / n >= self.min_support
        }

    def _generate_candidates(
        self, freq_prev: dict[frozenset, float], k: int
    ) -> list[frozenset]:
        """
        Generate candidate k-itemsets from frequent (k-1)-itemsets.
        Two (k-1)-itemsets are joined when they share exactly k-2 items.
        Each resulting candidate is then pruned if any of its (k-1)-subsets
        is not in freq_prev.
        
        Args:
            freq_prev: Dictionary of frequent (k-1)-itemsets with their support.
            k: Size of candidates to generate.
            
        Returns:
            List of candidate k-itemsets (as frozensets).
        """
        prev_list = sorted([sorted(fs) for fs in freq_prev])
        freq_set = set(freq_prev.keys())
        candidates = []

        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                # Join: both lists share the first k-2 elements
                if prev_list[i][: k - 2] == prev_list[j][: k - 2]:
                    candidate = frozenset(prev_list[i]) | frozenset(prev_list[j])
                    if len(candidate) == k and self._has_frequent_subsets(
                        candidate, freq_set, k
                    ):
                        candidates.append(candidate)
                else:
                    break  # sorted order; no more shared prefix possible

        return candidates

    @staticmethod
    def _has_frequent_subsets(
        candidate: frozenset, freq_set: set[frozenset], k: int
    ) -> bool:
        """Return True if every (k-1)-subset of candidate is frequent."""
        for subset in combinations(candidate, k - 1):
            if frozenset(subset) not in freq_set:
                return False
        return True

    def _count_and_filter(
        self, encoded: list[frozenset], candidates: list[frozenset]
    ) -> dict[frozenset, float]:
        """Count candidate support over the database and apply min_support."""
        counts: dict[frozenset, int] = defaultdict(int)
        for transaction in encoded:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    counts[candidate] += 1

        n = self._n_transactions
        return {
            itemset: count / n
            for itemset, count in counts.items()
            if count / n >= self.min_support
        }

from itertools import combinations
from collections import defaultdict
import time
import pandas as pd


class ECLAT:
    """
    The ECLAT (**Equivalence Class Clustering and bottom-up Lattice Traversal**) algorithm is a depth-first search algorithm that uses a vertical database
    structure. Rather than explicitly listing all transactions, each item is associated 
    with its coverage (or list of transactions containing that item). The intersection 
    approach is used to calculate the support of itemsets. This algorithm is particularly 
    efficient for small datasets and requires less space and time than the Apriori 
    algorithm to generate frequent patterns.

    For details on algorithm consult 
    `research paper <https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2000%20-%20IEEETKDE%20-%20Zaki%20-%20(Eclat)%20ScalableAlgorithms%20for%20Association%20Mining%20.pdf>`_.

    Args:
        min_support (float): Minimum support threshold (between 0 and 1) for frequent itemsets.
        min_confidence (float | None): Optional confidence threshold in (0, 1]
            kept for API consistency with rule-generation workflows.
        
        
    Examples:

    >>> transactions = [
    ...     {'bread', 'milk', 'butter'},
    ...     {'bread', 'jam', 'eggs'},
    ...     {'milk', 'butter', 'cheese'},
    ...     {'bread', 'milk', 'butter', 'cheese'},
    ...     {'bread', 'jam', 'milk'}
    ... ]
    >>> from ifri_mini_ml_lib.association_rules import ECLAT
    >>> eclat = ECLAT(min_support=0.4, min_confidence=0.6)
    >>> eclat.fit(transactions)
    <ifri_mini_ml_lib.association_rules.eclat.ECLAT object>
    >>> frequent_itemsets = eclat.get_frequent_itemsets()
    >>> print(frequent_itemsets.head(3))
    """

    def __init__(self, min_support: float = 0.5, min_confidence: float | None = None):
        """
        Initialize the ECLAT algorithm.
        
        Args:
            min_support (float): Minimum support threshold in [0, 1]. Defaults to 0.5.
            min_confidence (float | None): Optional confidence threshold in (0, 1].
                Defaults to None.
        
        Raises:
            ValueError: If min_support is not in [0, 1] or min_confidence is invalid.
        """
        if not 0.0 <= min_support <= 1.0:
            raise ValueError("min_support must be between 0 and 1.")
        if min_confidence is not None and not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1].")
        
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._frequent_itemsets: dict[frozenset, float] = {}
        self._n_transactions: int = 0
        self._execution_time: float = 0.0

    def fit(self, transactions: list[list]) -> "ECLAT":
        """
        Mine frequent itemsets from a list of transactions using vertical data format.

        Args:
            transactions (list[list]): Each inner list is one transaction containing hashable items.

        Returns:
            ECLAT: The current instance for method chaining.
        
        Raises:
            ValueError: If transactions list is empty.
        """
        start_time = time.time()
        if not transactions:
            raise ValueError("transactions must not be empty.")

        self._n_transactions = len(transactions)
        self._frequent_itemsets = {}

        # Build vertical representation: item -> set of transaction indices
        vertical: dict = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                vertical[frozenset([item])].add(tid)

        # Filter single items by min_support
        min_count = self.min_support * self._n_transactions
        freq_1 = {
            item: tids
            for item, tids in vertical.items()
            if len(tids) >= min_count
        }

        # Record 1-itemsets
        for item, tids in freq_1.items():
            self._frequent_itemsets[item] = len(tids) / self._n_transactions

        # Recursive depth-first enumeration
        self._eclat_recursive(list(freq_1.items()))

        self._execution_time = time.time() - start_time
        return self

    def get_frequent_itemsets(self) -> pd.DataFrame:
        """
        Return mined frequent itemsets as a DataFrame.

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

    def _eclat_recursive(self, prefix_class: list[tuple]) -> None:
        """
        Recursively extend each itemset in prefix_class by intersecting
        tidsets with every subsequent item in the same equivalence class.

        Args: 
            prefix_class : list of (frozenset, set) pairs
            Each pair is (itemset, tidset).
            
        Returns:
            None
        """
        min_count = self.min_support * self._n_transactions

        for i in range(len(prefix_class)):
            itemset_i, tids_i = prefix_class[i]
            next_level = []

            for j in range(i + 1, len(prefix_class)):
                itemset_j, tids_j = prefix_class[j]

                # Compute support of the union via tidset intersection
                new_tids = tids_i & tids_j
                if len(new_tids) >= min_count:
                    new_itemset = itemset_i | itemset_j
                    self._frequent_itemsets[new_itemset] = (
                        len(new_tids) / self._n_transactions
                    )
                    next_level.append((new_itemset, new_tids))

            if next_level:
                self._eclat_recursive(next_level)


# Backward-compatible alias (keeps existing imports working).
Eclat = ECLAT

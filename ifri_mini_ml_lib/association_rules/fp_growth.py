from collections import defaultdict
import time
import pandas as pd
class _FPNode:
    """
    Node of the FP-Tree structure.

    Attributes:
        item: The item represented by this node.
        count: The number of occurrences of this item.
        parent: The parent node in the tree.
        children: Dictionary of child nodes.
        node_link: Link to the next node containing the same item.
    """

    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item, count: int = 0, parent=None):
        self.item = item          # None for the root
        self.count = count
        self.parent = parent
        self.children: dict = {}  # item -> _FPNode
        self.node_link = None     # next node with the same item (header table)

    def increment(self, count: int = 1) -> None:
        """Increment node count."""
        self.count += count


class _FPTree:
    """
    FP-Tree structure with its header table.

    The header table maps each frequent item to
    [support_count, first_node_in_chain].
    """

    def __init__(self):
        self.root = _FPNode(item=None, count=0)
        self.header: dict = {}

    def insert_transaction(self, transaction: list, count: int = 1) -> None:
        """Insert an ordered transaction (most-frequent item first)."""
        node = self.root
        for item in transaction:
            if item in node.children:
                node.children[item].increment(count)
            else:
                child = _FPNode(item=item, count=count, parent=node)
                node.children[item] = child
                self._update_header(item, child)
            node = node.children[item]

    def _update_header(self, item, new_node: _FPNode) -> None:
        """Link new_node into the header table chain for item."""
        if item not in self.header:
            self.header[item] = [new_node.count, new_node]
            return

        self.header[item][0] += new_node.count
        current = self.header[item][1]
        while current.node_link is not None:
            current = current.node_link
        current.node_link = new_node


class FPGrowth:
    """
    Frequent itemset mining via FP-Growth.

    For more details on the algorithm, consult the original paper and a
    practical guide:
    - Han, J., Pei, J., & Yin, Y. (2000). DOI: https://doi.org/10.1145/335191.335372
    - Practical tutorial and examples: https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/

    The algorithm is performed in two main passes over the dataset:
    1- Count item frequencies and keep only frequent items.
    2- Build a compact FP-Tree with transactions sorted by descending global support.

    Mining then proceeds recursively by constructing conditional pattern bases
    and conditional FP-Trees, which greatly reduces candidate explosion compared
    with Apriori-style generation.

    Args:
        min_support (float): Minimum support threshold in (0, 1].
        min_confidence (float | None): Optional confidence threshold in (0, 1]
            kept for API consistency with rule-generation workflows. Rule mining
            itself is handled in the AssociationRules class.

    Attributes:
        min_support (float): User-defined minimum support.
        min_confidence (float | None): Optional confidence parameter for API parity.
        execution_time (float): Execution time in seconds for the latest fit call.

    Example:
        >>> transactions = [
        ...     {'bread', 'milk', 'butter'},
        ...     {'bread', 'jam', 'eggs'},
        ...     {'milk', 'butter', 'cheese'},
        ... ]
        >>> model = FPGrowth(min_support=0.4, min_confidence=0.6)
        >>> model.fit(transactions)
        >>> model.get_frequent_itemsets().head()
    """

    def __init__(self, min_support: float = 0.5, min_confidence: float | None = None):
        """
        Initialize the FP-Growth algorithm.

        Args:
            min_support (float): Minimum support threshold in (0, 1]. Defaults to 0.5.
            min_confidence (float | None): Optional confidence threshold in (0, 1].
                Defaults to None.

        Raises:
            ValueError: If min_support is not in (0, 1] or min_confidence is invalid.
        """
        if not 0.0 < min_support <= 1.0:
            raise ValueError("min_support must be in (0, 1].")
        if min_confidence is not None and not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1].")

        self.min_support = min_support
        self.min_confidence = min_confidence
        self._frequent_itemsets: dict[frozenset, float] = {}
        self._n_transactions: int = 0
        self._execution_time: float = 0.0

    def fit(self, transactions: list[list]) -> "FPGrowth":
        """
        Mine frequent itemsets from a list of transactions.

        This method stores frequent itemsets and their support in
        self._frequent_itemsets as {frozenset(items): support}.

        Args:
            transactions (list[list]): Each inner list is one transaction containing hashable items.

        Returns:
            FPGrowth: The current instance for method chaining.

        Raises:
            ValueError: If transactions list is empty.
        """
        start_time = time.time()
        if not transactions:
            raise ValueError("transactions must not be empty.")

        self._n_transactions = len(transactions)
        self._frequent_itemsets = {}
        min_count = self.min_support * self._n_transactions

        # Pass 1: count single-item frequencies
        item_counts: dict = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        freq_items = {
            item: count
            for item, count in item_counts.items()
            if count >= min_count
        }

        if not freq_items:
            self._execution_time = time.time() - start_time
            return self

        # Pass 2: build FP-Tree
        sort_key = lambda x: (-freq_items[x], str(x))
        tree = _FPTree()
        for transaction in transactions:
            filtered = sorted(
                [item for item in transaction if item in freq_items],
                key=sort_key,
            )
            if filtered:
                tree.insert_transaction(filtered)

        # Mine the tree
        self._mine_tree(tree, frozenset(), min_count)

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


    def _mine_tree(
        self, tree: _FPTree, prefix: frozenset, min_count: float
    ) -> None:
        """
        Recursively mine an FP-Tree with the given prefix.

        Notes:
            Items are processed in ascending support order to follow the classic
            FP-Growth pattern-growth strategy from least frequent suffixes.

        Args:
            tree (_FPTree): FP-Tree to mine.
            prefix (frozenset): Current prefix itemset.
            min_count (float): Minimum support count threshold.

        Returns:
            None
        """
        items = sorted(tree.header.keys(), key=lambda x: tree.header[x][0])

        for item in items:
            support_count = tree.header[item][0]
            new_prefix = prefix | frozenset([item])
            self._frequent_itemsets[new_prefix] = (
                support_count / self._n_transactions
            )

            cond_patterns = self._build_cond_patterns(tree, item)

            cond_tree = _FPTree()
            for pattern, count in cond_patterns:
                cond_tree.insert_transaction(pattern, count)

            cond_tree.header = {
                k: v
                for k, v in cond_tree.header.items()
                if v[0] >= min_count
            }

            if cond_tree.header:
                self._mine_tree(cond_tree, new_prefix, min_count)

    @staticmethod
    def _build_cond_patterns(tree: _FPTree, item) -> list[tuple]:
        """
        Collect all prefix paths (conditional pattern base) for an item.

        Args:
            tree (_FPTree): Source FP-Tree.
            item: Item for which to build conditional patterns.

        Returns:
            list[tuple]: List of tuples (path, count), where path is ordered
                from root to leaf parent for a node linked to item.
        """
        patterns = []
        node = tree.header[item][1]
        while node is not None:
            path = []
            parent = node.parent
            while parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                patterns.append((path[::-1], node.count))
            node = node.node_link
        return patterns

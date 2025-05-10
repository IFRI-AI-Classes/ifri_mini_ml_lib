from itertools import combinations, chain
import time


class Eclat:
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
        min_confidence (float): Minimum confidence threshold (between 0 and 1) for association rules.
        
    Examples:

    >>> transactions = [
    ...     {'bread', 'milk', 'butter'},
    ...     {'bread', 'jam', 'eggs'},
    ...     {'milk', 'butter', 'cheese'},
    ...     {'bread', 'milk', 'butter', 'cheese'},
    ...     {'bread', 'jam', 'milk'}
    ... ]
    >>> from ifri_mini_ml_lib.association_rules import Eclat
    >>> eclat = Eclat(min_support=0.4, min_confidence=0.6)
    >>> eclat.fit(transactions) # Frequents itemsets + Rules generation
    <ifri_mini_ml_lib.association_rules.eclat.Eclat object>
    >>> frequent_itemsets = eclat.get_frequent_itemsets()
    >>> # Displaying frequent itemsets of size 1
    >>> for itemset, tidset in frequent_itemsets[1].items():
    ...     item = list(itemset)[0]
    ...     support = len(tidset) / len(transactions)
    ...     print(f"Item: {item}, Support: {support:.2f}")
    Item: bread, Support: 0.80
    Item: milk, Support: 0.80
    Item: butter, Support: 0.60
    >>> rules = eclat.get_rules()
    >>> # Displaying some association rules
    >>> if rules:
    ...     for rule in rules[:2]:
    ...         print(f"{set(rule['antecedent'])} -> {set(rule['consequent'])}, "
    ...               f"Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}")
    {'milk'} -> {'bread'}, Confidence: 0.75, Lift: 0.94
    {'bread'} -> {'milk'}, Confidence: 0.75, Lift: 0.94
    """

    def __init__(self, min_support: float, min_confidence:float):
        if not 0 <= min_support <= 1:
            raise ValueError("Minimum support must be between 0 and 1")
        if not 0 <= min_confidence <= 1:
            raise ValueError("Minimum confidence must be between 0 and 1")

        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.frequent_TIDsets = {}
        self.rules_ = []
        self.n_transactions = 0

    def fit(self, transactions):
        """
        Main method for learning frequent itemsets.
        
        Args:
            transactions: List of transactions (each transaction is a set of items)

        Returns:
            self: The current instance for method chaining
        """
        start_time = time.time()
        
        # Check the conformity of the input data format
        if isinstance(transactions, list):
            transactions_list = transactions
        else:
            raise TypeError("Data format not respected! Only List[set] format is accepted.")
        
        self.n_transactions = len(transactions_list)
        
        print(f"\nApplying Eclat algorithm with:")
        print(f"- Minimum support: {self.min_support} ({self.min_support*100}%)")
        print(f"- Minimum confidence: {self.min_confidence} ({self.min_confidence*100}%)")
        if transactions_list:
            print(f"Number of valid transactions: {self.n_transactions}")
        else:
            return self

        # Find frequent itemsets
        self._fit_eclat(transactions_list)

        # Generate association rules
        self._generate_rules()

        elapsed_time = time.time() - start_time

        print(f"\nExecution time: {elapsed_time:.2f} seconds")
        return self

    def _fit_eclat(self, transactions):
        """
        Implementation of the frequent itemset extraction phase.
        
        Args:
            transactions: List of transactions (each transaction is a set of items)
        """
        # Get all unique items in the transactions
        all_items = set(chain(*transactions))
        
        # Store frequent items of size 1
        self.frequent_itemsets[1] = self.get_single_items_TIDset(all_items, transactions)
        
        # Generate itemsets of increasing size
        k = 1
        while self.frequent_itemsets.get(k):            
            k += 1
            candidates = self._generate_candidates(k)
            
            if not candidates:
                break
                
            # Calculate TIDsets for the new candidates
            level_frequent = {}
            for candidate in candidates:
                subsets = [frozenset(subset) for subset in combinations(candidate, k-1)]
                if all(subset in self.frequent_itemsets[k-1] for subset in subsets):
                    # Intersection of TIDsets of subsets
                    TIDset = set.intersection(*[self.frequent_itemsets[k-1][subset] for subset in subsets])
                    
                    # Check support
                    if len(TIDset) / self.n_transactions >= self.min_support:
                        level_frequent[candidate] = TIDset
            
            if level_frequent:
                self.frequent_itemsets[k] = level_frequent
            else:
                break

    def get_single_items_TIDset(self, items, transactions):
        """Build TIDsets for individual items"""
        single_items_TIDsets = {}
        for item in items:
            TIDset = set()
            for tid, transaction in enumerate(transactions):
                if item in transaction:
                    TIDset.add(tid)
            
            # Check if the item meets the minimum support
            if len(TIDset) / self.n_transactions >= self.min_support:
                single_items_TIDsets[frozenset([item])] = TIDset

        return single_items_TIDsets
    
    def _generate_candidates(self, k):
        """
        Generate candidates of size k from frequent itemsets of size k-1.
        
        Args:
            k: Size of candidates to generate
            
        Returns:
            Set of generated candidates
        """
        candidates = set()
        prev_frequent = self.frequent_itemsets.get(k-1, {})
        
        for itemset1, itemset2 in combinations(prev_frequent.keys(), 2):
            # Join itemsets having k-2 elements in common
            union = itemset1.union(itemset2)
            if len(union) == k:
                # Check that all subsets of size k-1 are frequent
                if all(frozenset(subset) in prev_frequent for subset in combinations(union, k-1)):
                    candidates.add(union)
        
        return candidates

    def _generate_rules(self):
        """
        Generate association rules from frequent itemsets.
        """
        
        # Go through all frequent itemsets of size > 1
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
                
            for itemset, TIDset in self.frequent_itemsets[k].items():
                itemset_support = len(TIDset) / self.n_transactions
                
                # Generate all possible rules from this itemset
                for i in range(1, k):
                    for antecedent_items in combinations(itemset, i):
                        antecedent = frozenset(antecedent_items)
                        consequent = itemset.difference(antecedent)
                        
                        # Calculate confidence
                        antecedent_support = len(self.frequent_itemsets[len(antecedent)][antecedent]) / self.n_transactions
                        confidence = itemset_support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            # Calculate lift
                            consequent_support = len(self.frequent_itemsets[len(consequent)][consequent]) / self.n_transactions
                            lift = confidence / consequent_support
                            
                            # Add the rule to the list
                            self.rules_.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': itemset_support,
                                'confidence': confidence,
                                'lift': lift
                            })


    def get_frequent_itemsets(self):
        """
        Retrieve the discovered frequent itemsets.
        
        Returns:
            dict: Dictionary of frequent itemsets where keys are sizes
                and values are sets of itemsets
        """
        return self.frequent_itemsets
    
    def get_rules(self):
        """
        Accessor to retrieve the generated association rules.
        
        Returns:
            List of association rules
        """
        return self.rules_

from .utils import DataAdapter
from .metrics import support, confidence, lift
from itertools import chain, combinations
from collections import defaultdict
import time

class Apriori:
    """
    The Apriori algorithm is used to discover interesting association rules
    in large transactional datasets. For more details: 
    `Agrawal, R., & Srikant, R. (1994, September) <http://www.vldb.org/conf/1994/P487.PDF>`_

    Args:
        min_support(float): Minimum support threshold for considering an itemset
        min_confidence(float): Minimum confidence threshold for a rule
        
    Examples:

    >>> transactions = [
    ...     {'bread', 'milk', 'butter'},
    ...     {'bread', 'jam', 'eggs'},
    ...     {'milk', 'butter', 'cheese'},
    ...     {'bread', 'milk', 'butter', 'cheese'},
    ...     {'bread', 'jam', 'milk'}
    ... ]
    >>> from ifri_mini_ml_lib.association_rules import Apriori
    >>> apriori = Apriori(min_support=0.4, min_confidence=0.6)
    >>> apriori.fit(transactions) # Frequents itemsets + Rules generation
    <ifri_mini_ml_lib.association_rules.apriori.Apriori object>
    >>> frequent_itemsets = apriori.get_frequent_itemsets()
    >>> # Displaying frequent itemsets of size 1
    >>> for item in frequent_itemsets[1]:
    ...     print(f"Item: {list(item)[0]}")
    Item: bread
    Item: milk
    Item: butter
    >>> rules = apriori.get_rules()
    >>> # Displaying some association rules
    >>> if rules:
    ...     for rule in rules[:2]:
    ...         print(f"{set(rule['antecedent'])} -> {set(rule['consequent'])}, "
    ...               f"Support: {rule['support']:.2f}, Confidence: {rule['confidence']:.2f}")
    {'milk'} -> {'bread'}, Support: 0.60, Confidence: 0.75
    {'bread'} -> {'milk'}, Support: 0.60, Confidence: 0.75
    """
    def __init__(self, min_support: float, min_confidence: float):
        
        if not 0 <= min_support <= 1:
            raise ValueError("Minimum support must be between 0 and 1")
        if not 0 <= min_confidence <= 1:
            raise ValueError("Minimum confidence must be between 0 and 1")
        
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets_ = {}
        self.rules_ = []

    def fit(self, transactions: list):
        """
        Main method for learning frequent itemsets and rules.
        
        Args:
            transactions: Input data (list[set])
            
        Returns:
            self: The current instance for method chaining
        """
        start_time = time.time()
        
        # Check the conformity of the input data format
        if isinstance(transactions, list):
            transactions_list = DataAdapter._convert_from_list(transactions)
        else:
            raise TypeError("Data format not respected! Only List[set] format is accepted.")
        
        print(f"\nApplying Apriori algorithm with:")
        print(f"- Minimum support: {self.min_support} ({self.min_support*100}%)")
        print(f"- Minimum confidence: {self.min_confidence} ({self.min_confidence*100}%)")

        print(f"Number of valid transactions: {len(transactions_list)}")
        if transactions_list:
            print(f"Example transaction: {list(transactions_list[0])[:5]}...")
        else:
            return self

        # Generate frequent itemsets
        self._fit_apriori(transactions_list)
        # Generate association rules
        self._generate_rules(transactions_list)

        elapsed_time = time.time() - start_time

        print(f"\nExecution time: {elapsed_time:.2f} seconds")
        return self
    
    def _fit_apriori(self, transactions: list[set]):
        """
        Extraction of k-frequent itemsets

        ## Steps:
            1. Extract all unique items
            2. Find 1-itemsets that are frequent
            3. Iteratively generate itemsets of increasing size
        """
        # Get unique items
        items = set(chain(*transactions))
        # Get 1-itemsets that are frequent
        self.frequent_itemsets_ = {
            1: self._get_one_itemsets(items, transactions)
        }
        
        # Generate itemsets of increasing size
        size = 1
        while True:
            size += 1
            candidates = self._generate_candidates(self.frequent_itemsets_[size-1])
            frequent = self._prune_candidates(candidates, transactions)

            # Stop if no frequent item is found
            if not frequent: 
                break
            # Store frequent itemsets of this size
            self.frequent_itemsets_[size] = frequent

    def _get_one_itemsets(self, items: set, transactions: list[set]):
        """
        Computes the frequent 1-itemsets (individual items).
        
        Args:
            items: Set of all unique items
            transactions: List of transactions
        
        Returns:
            Set of frequent items that satisfy the minimum support
        """
        # Occurrence of each unique item in the database
        items_counts = defaultdict(int)
        for t in transactions:
            for i in items:
                if i in t:
                    items_counts[frozenset([i])] +=1
        
        n_trans = len(transactions)

        return {i for i, count in items_counts.items()
                if count/n_trans >= self.min_support}

    def _generate_candidates(self, previous_itemsets: set):
        """
        Generates new candidates by combining previous itemsets.
        Uses the optimized "join-and-prune" approach that only combines
        itemsets sharing the same first k-1 elements.
        
        Args:
            previous_itemsets: Itemsets of the previous size k
        
        Returns:
            New candidates of size k + 1
        """
        candidates = set()
        previous_list = list(previous_itemsets)
        k = len(list(previous_list[0])) if previous_list else 0
        
        # Join phase
        for i in range(len(previous_list)):
            for j in range(i+1, len(previous_list)):
                # Convert to list to compare elements by index
                items1 = sorted(list(previous_list[i]))
                items2 = sorted(list(previous_list[j]))
                
                # Check if the first k-1 elements are identical
                if items1[:k-1] == items2[:k-1]:
                    # Create a new candidate itemset
                    new_candidate = frozenset(previous_list[i] | previous_list[j])
                    
                    # Prune phase - all subsets must be frequent
                    should_add = True
                    for subset in combinations(new_candidate, k):
                        if frozenset(subset) not in previous_itemsets:
                            should_add = False
                            break
                    
                    if should_add:
                        candidates.add(new_candidate)
        
        return candidates

    def _prune_candidates(self, candidates: set, transactions: list[set]):
        """
        Filters candidates according to minimum support in an optimized way.
        
        Args:
            candidates: Set of candidates to test
            transactions: List of transactions
        
        Returns:
            Frequent itemsets among the candidates
        """
        if not candidates:
            return set()
            
        return {c for c in candidates if support(c, transactions) >= self.min_support}
    
    def _generate_rules(self, transactions: list[set]):
        """
        Generates association rules from frequent itemsets.
        Calculates confidence and lift for each rule.
        
        Args:
            transactions: List of transactions
        """

        # Go through frequent itemsets ignoring those of size 1
        for itemset in chain(*(self.frequent_itemsets_.values())):
            if len(itemset) < 2:
                continue

            # Generate all possible rule combinations
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    # Calculate metrics
                    conf = confidence(antecedent, consequent, transactions)
                    
                    if conf >= self.min_confidence:
                        rule_support = support(itemset, transactions)
                        rule_lift = lift(antecedent, consequent, transactions)
                        self.rules_.append({
                            'antecedent': antecedent, 
                            'consequent': consequent, 
                            'support': rule_support,
                            'confidence': conf,
                            'lift': rule_lift
                        })

    def get_frequent_itemsets(self):
        """
        Retrieve the discovered frequent itemsets.
        
        Returns:
            dict: Dictionary of frequent itemsets where keys are sizes
                and values are sets of itemsets
        """
        return self.frequent_itemsets_

    def get_rules(self):
        """
        Retrieve the generated association rules.
        
        Returns:
            List of association rules
        """
        return self.rules_


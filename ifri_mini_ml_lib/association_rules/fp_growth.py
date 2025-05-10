import time
from collections import defaultdict
from itertools import combinations


class FPNode:
    """
    Node of the FP-Tree structure.
    
    Attributes:
        item: The item represented by this node
        count: The number of occurrences of this item
        parent: The parent node in the tree
        children: Dictionary of child nodes
        node_link: Link to the next node containing the same item
    """
    def __init__(self, item, count=1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None

    def increment(self, count=1):
        """Increments the node counter."""
        self.count += count


class FPGrowth:
    """
    FP-Growth is an efficient algorithm for extracting frequent itemsets that
    constructs a compact data structure (FP-Tree) and extracts patterns without
    generating candidates, unlike Apriori or Eclat. For more details consult the following
    `link <https://dl.acm.org/doi/10.1145/342009.335372>`_.
    
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
        >>> from ifri_mini_ml_lib.association_rules import FPGrowth
        >>> fp_growth = FPGrowth(min_support=0.4, min_confidence=0.6)
        >>> fp_growth.fit(transactions) # Frequents itemsets + Rules generation
        <ifri_mini_ml_lib.association_rules.fp_growth.FPGrowth object>
        >>> frequent_itemsets = fp_growth.get_frequent_itemsets()
        >>> for itemset_data in frequent_itemsets[:3]:
        ...     print(f"Itemset: {set(itemset_data['itemset'])}, Support: {itemset_data['support']:.2f}")
        Itemset: {'bread'}, Support: 0.80
        Itemset: {'milk'}, Support: 0.80
        Itemset: {'butter'}, Support: 0.60
        >>> rules = fp_growth.get_rules()
        >>> # Displaying some association rules
        >>> if rules:
        ...     for rule in rules[:2]:
        ...         print(f"{set(rule['antecedent'])} -> {set(rule['consequent'])}, "
        ...               f"Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}")
        {'milk'} -> {'bread'}, Confidence: 0.75, Lift: 0.94
        {'bread'} -> {'milk'}, Confidence: 0.75, Lift: 0.94

    """
    def __init__(self, min_support: float, min_confidence: float):
        
        if not 0 <= min_support <= 1:
            raise ValueError("Minimum support must be between 0 and 1")
        if not 0 <= min_confidence <= 1:
            raise ValueError("Minimum confidence must be between 0 and 1")
        
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets_dict = {}
        self.rules_ = []
        self.header_table = {}
        self.n_transactions = 0
        
    def fit(self, transactions):
        """
        Builds the FP-Tree and extracts frequent itemsets.
        
        Args:
            transactions: List of transactions (each transaction is a set of items)
            
        Returns:
            self: The current instance for method chaining
        """
        start_time = time.time()
        
        if not isinstance(transactions, list):
            raise TypeError("Data format not respected! Only List[set] format is accepted.")
            
        self.n_transactions = len(transactions)
        self.min_support_count = int(self.min_support * self.n_transactions)
        
        print(f"\nApplying FP-Growth algorithm with:")
        print(f"- Minimum support: {self.min_support} ({self.min_support*100}%)")
        print(f"- Minimum support count: {self.min_support_count}")
        print(f"- Minimum confidence: {self.min_confidence} ({self.min_confidence*100}%)")
        print(f"Number of valid transactions: {self.n_transactions}")

        if len(transactions) == 0:
            return self
        
        # Step 1: Count the frequency of individual items
        item_counter = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counter[item] += 1
        
        # Filter frequent items and create the header table
        self.header_table = {
            item: {
                'count': count, 
                'head': None
            } 
            for item, count in item_counter.items() 
            if count >= self.min_support_count
        }
        
        # Reset frequent itemsets
        self.frequent_itemsets_dict = {}
        
        # If no frequent item is found, return
        if not self.header_table:
            elapsed_time = time.time() - start_time
            print(f"\nExecution time: {elapsed_time:.2f} seconds")
            print("No frequent itemset found.")
            return self
        
        # Sort items by decreasing frequency
        self.ordered_items = sorted(
            self.header_table.keys(), 
            key=lambda x: (-self.header_table[x]['count'], x)
        )
        
        # Step 2: Build the FP-Tree
        self.root = FPNode(None, 0)
        
        # Insert each transaction into the tree
        for transaction in transactions:
            # Filter non-frequent items and sort
            items = [item for item in transaction if item in self.header_table]
            items.sort(key=lambda x: (-self.header_table[x]['count'], x))
            
            if items:
                self._insert_tree(items, self.root, 1)
        
        # Step 3: Extract frequent itemsets
        self._mine_tree(self.root, set(), self.min_support_count)

        # Step 4: Generate rules
        self._generate_rules()
        
        elapsed_time = time.time() - start_time
        print(f"\nExecution time: {elapsed_time:.2f} seconds")
        print(f"Number of frequent itemsets found: {len(self.frequent_itemsets_dict)}")
        
        return self
    
    def _insert_tree(self, items, node, count=1):
        """
        Inserts a transaction into the FP-Tree.
        
        Args:
            items: List of items in the transaction (sorted by frequency)
            node: Current node in the tree
            count: Number of occurrences to add
        """
        if not items:
            return
        
        item = items[0]
        
        # If the item is already a child of the current node
        if item in node.children:
            node.children[item].increment(count)
        else:
            # Create a new node
            new_node = FPNode(item, count, node)
            node.children[item] = new_node
            
            # Update the header table (links between nodes)
            if self.header_table[item]['head'] is None:
                self.header_table[item]['head'] = new_node
            else:
                self._update_header_link(item, new_node)
        
        # Recursive insertion for remaining items
        self._insert_tree(items[1:], node.children[item], count)
    
    def _update_header_link(self, item, target_node):
        """Update links between nodes containing the same item."""
        current = self.header_table[item]['head']
        while current.node_link is not None:
            current = current.node_link
        current.node_link = target_node
    
    def _find_prefix_path(self, node):
        """
        Finds the prefix path for a given node.
        
        Args:
            node: Node for which to find the prefix path
            
        Returns:
            List of prefix paths with their support
        """
        prefix_paths = []
        
        while node is not None and node.parent is not None and node.parent.item is not None:
            prefix_paths.append(node.parent.item)
            node = node.parent
            
        return prefix_paths
    
    def _mine_tree(self, node, prefix, min_support_count):
        """
        Recursively extracts frequent itemsets from the FP-Tree.
        
        Args:
            node: Root node of the FP-Tree
            prefix: Current prefix (itemset under construction)
            min_support_count: Minimum support in absolute count
        """
        # Traverse items in reverse order of frequency to
        # generate smaller trees during extraction (optimization)
        for item in reversed(self.ordered_items):
            new_prefix = prefix.copy()
            new_prefix.add(item)
            new_itemset = frozenset(new_prefix)
            
            # Add this itemset to the list of frequent itemsets if it doesn't already exist
            support = self.header_table[item]['count']
            support_ratio = support / self.n_transactions
            
            self.frequent_itemsets_dict[new_itemset] = {
                'support_count': support,
                'support': support_ratio
            }
            
            # Build the conditional pattern base
            conditional_pattern_base = []
            
            # Traverse all nodes containing this item
            node_ref = self.header_table[item]['head']
            while node_ref is not None:
                prefix_path = self._find_prefix_path(node_ref)
                if prefix_path:
                    conditional_pattern_base.append((prefix_path, node_ref.count))
                node_ref = node_ref.node_link
            
            # Build the conditional FP-Tree if paths exist
            if conditional_pattern_base:
                conditional_tree = FPNode(None, 0)
                
                # Recalculate supports in this conditional base
                cond_item_counts = defaultdict(int)
                for path, count in conditional_pattern_base:
                    for path_item in path:
                        cond_item_counts[path_item] += count
                
                # Filter frequent items in this conditional base
                frequent_items = {item: count for item, count in cond_item_counts.items() 
                                if count >= min_support_count}
                
                # If frequent items exist, build the conditional tree
                if frequent_items:
                    # Insert paths into the conditional tree
                    for path, count in conditional_pattern_base:
                        # Filter and sort path items
                        filtered_path = [p for p in path if p in frequent_items]
                        filtered_path.sort(key=lambda x: (-frequent_items[x], x))
                        
                        if filtered_path:
                            # Create a mini header table for this conditional tree
                            header_table = {item: {'count': 0, 'head': None} for item in frequent_items}
                            self._insert_conditional_tree(filtered_path, conditional_tree, count, header_table)
                    
                    # Recursively extract itemsets from this conditional tree
                    for cond_item in sorted(frequent_items, key=lambda x: (-frequent_items[x], x)):
                        newer_prefix = new_prefix.copy()
                        newer_prefix.add(cond_item)
                        newer_itemset = frozenset(newer_prefix)
                        
                        # Add this new frequent itemset
                        cond_support = frequent_items[cond_item]
                        cond_support_ratio = cond_support / self.n_transactions
                        
                        self.frequent_itemsets_dict[newer_itemset] = {
                            'support_count': cond_support,
                            'support': cond_support_ratio
                        }
        
    def _insert_conditional_tree(self, items, node, count, header_table):
        """
        Inserts a path into a conditional tree.
        
        Args:
            items: List of items in the path
            node: Current node in the tree
            count: Support of the path
            header_table: Header table for this conditional tree
        """
        if not items:
            return
        
        item = items[0]
        
        # Update the count in the header table
        header_table[item]['count'] += count
        
        # Insert into the tree
        if item in node.children:
            node.children[item].increment(count)
        else:
            new_node = FPNode(item, count, node)
            node.children[item] = new_node
            
            # Update the header table
            if header_table[item]['head'] is None:
                header_table[item]['head'] = new_node
            else:
                current = header_table[item]['head']
                while current.node_link:
                    current = current.node_link
                current.node_link = new_node
        
        # Recursive insertion
        self._insert_conditional_tree(items[1:], node.children[item], count, header_table)
    
    def _generate_rules(self):
        """
        Generates association rules from frequent itemsets.

        Returns:
            List of association rules
        """
        
        # Examine all itemsets of size > 1
        for itemset, item_data in self.frequent_itemsets_dict.items():
            if len(itemset) < 2:
                continue
                
            # For each possible subset as antecedent
            for i in range(1, len(itemset)):
                for antecedent_items in self._subsets(itemset, i):
                    antecedent = frozenset(antecedent_items)
                    consequent = itemset - antecedent
                    
                    if antecedent in self.frequent_itemsets_dict and consequent in self.frequent_itemsets_dict:
                        # Calculate confidence and lift
                        confidence = item_data['support'] / self.frequent_itemsets_dict[antecedent]['support']
                        lift = confidence / self.frequent_itemsets_dict[consequent]['support']
                        
                        if confidence >= self.min_confidence:
                            self.rules_.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': item_data['support'],
                                'confidence': confidence,
                                'lift': lift
                            })
    
    def _subsets(self, s, n):
        """Generates all subsets of size n of a set."""
        return [set(combo) for combo in combinations(s, n)]

    def get_frequent_itemsets(self):
        """
        Retrieve the discovered frequent itemsets.
        
        Returns:
            List: List of frequent itemsets with their supports
        """
        # Convert dictionary to list for compatibility
        result = [{'itemset': list(itemset), **item_data} for itemset, item_data in self.frequent_itemsets_dict.items()]
        return result

    def get_rules(self):
        """
        Retrieve the generated association rules.
        
        Returns:
            List of association rules
        """
        return self.rules_

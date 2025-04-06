import time
from collections import defaultdict
from itertools import combinations


class FPNode:
    """
    Nœud de l'arbre FP-Tree.
    
    Attributes:
        item: L'item représenté par ce nœud
        count: Le nombre d'occurrences de cet item
        parent: Le nœud parent dans l'arbre
        children: Dictionnaire des nœuds enfants
        node_link: Lien vers le prochain nœud contenant le même item
    """
    def __init__(self, item, count=1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None

    def increment(self, count=1):
        """Incrémente le compteur du nœud."""
        self.count += count

    def display(self, indent=0):
        """Affiche le nœud et ses enfants de manière récursive."""
        if self.item is not None:
            print("  " * indent + f"{self.item}: {self.count}")
        for child in self.children.values():
            child.display(indent + 1)


class FPGrowth:
    """
    FP-Growth est un algorithme efficace pour l'extraction d'itemsets fréquents qui
    construit une structure de données compacte (FP-Tree) et extrait les motifs sans
    générer de candidats, contrairement à Apriori ou Eclat.
    
    Args:
        min_support (float): Seuil minimal de support (entre 0 et 1) pour les itemsets fréquents.
        min_confiance (float): Seuil minimal de confiance (entre 0 et 1) pour les règles d'association.
    """
    def __init__(self, min_support: float, min_confiance: float):
        
        if not 0 <= min_support <= 1:
            raise ValueError("Le support minimal doit être compris entre 0 et 1")
        if not 0 <= min_confiance <= 1:
            raise ValueError("La confiance minimale doit être comprise entre 0 et 1")
        
        self.min_support = min_support
        self.min_confiance = min_confiance
        self.frequent_itemsets_dict = {}
        self.rules_ = None
        self.header_table = {}
        self.n_transactions = 0
        
    def fit(self, transactions):
        """
        Construit le FP-Tree et extrait les itemsets fréquents.
        
        Args:
            transactions: Liste de transactions (chaque transaction est un ensemble d'items)
            
        Returns:
            self: L'instance courante pour chaînage des méthodes
        """
        start_time = time.time()
        
        if not isinstance(transactions, list):
            raise TypeError("Format de données non respecté! List[set] format uniquement accepté.")
            
        self.n_transactions = len(transactions)
        self.min_support_count = int(self.min_support * self.n_transactions)
        
        print(f"\nApplication de l'algorithme FP-Growth avec:")
        print(f"- Support minimum: {self.min_support} ({self.min_support*100}%)")
        print(f"- Support minimum en nombre: {self.min_support_count}")
        print(f"- Confiance minimum: {self.min_confiance} ({self.min_confiance*100}%)")
        print(f"Nombre de transactions valides : {self.n_transactions}")
        
        # Étape 1: Compter la fréquence des items individuels
        item_counter = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counter[item] += 1
        
        # Filtrer les items fréquents et créer la table d'en-tête
        self.header_table = {
            item: {
                'count': count, 
                'head': None
            } 
            for item, count in item_counter.items() 
            if count >= self.min_support_count
        }
        
        # Réinitialiser les itemsets fréquents
        self.frequent_itemsets_dict = {}
        
        # Si aucun item fréquent n'est trouvé, retourner
        if not self.header_table:
            elapsed_time = time.time() - start_time
            print(f"\nTemps d'exécution: {elapsed_time:.2f} secondes")
            print("Aucun itemset fréquent trouvé.")
            return self
        
        # Trier les items par fréquence décroissante
        self.ordered_items = sorted(
            self.header_table.keys(), 
            key=lambda x: (-self.header_table[x]['count'], x)
        )
        
        # Étape 2: Construire l'arbre FP
        self.root = FPNode(None, 0)
        
        # Insérer chaque transaction dans l'arbre
        for transaction in transactions:
            # Filtrer les items non fréquents et trier
            items = [item for item in transaction if item in self.header_table]
            items.sort(key=lambda x: (-self.header_table[x]['count'], x))
            
            if items:
                self._insert_tree(items, self.root, 1)
        
        # Étape 3: Extraire les itemsets fréquents
        self._mine_tree(self.root, set(), self.min_support_count)

        # Etape 4: Générer les règles
        self._generate_rules()
        
        elapsed_time = time.time() - start_time
        print(f"\nTemps d'exécution: {elapsed_time:.2f} secondes")
        print(f"Nombre d'itemsets fréquents trouvés: {len(self.frequent_itemsets_dict)}")
        
        return self
    
    def _insert_tree(self, items, node, count=1):
        """
        Insère une transaction dans le FP-Tree.
        
        Args:
            items: Liste des items de la transaction (triés par fréquence)
            node: Nœud courant dans l'arbre
            count: Nombre d'occurrences à ajouter
        """
        if not items:
            return
        
        item = items[0]
        
        # Si l'item est déjà un enfant du nœud courant
        if item in node.children:
            node.children[item].increment(count)
        else:
            # Créer un nouveau nœud
            new_node = FPNode(item, count, node)
            node.children[item] = new_node
            
            # Mettre à jour la table d'en-tête (liens entre nœuds)
            if self.header_table[item]['head'] is None:
                self.header_table[item]['head'] = new_node
            else:
                self._update_header_link(item, new_node)
        
        # Insertion récursive pour les items restants
        self._insert_tree(items[1:], node.children[item], count)
    
    def _update_header_link(self, item, target_node):
        """Mise à jour des liens entre nœuds contenant le même item."""
        current = self.header_table[item]['head']
        while current.node_link is not None:
            current = current.node_link
        current.node_link = target_node
    
    def _find_prefix_path(self, node):
        """
        Trouve le chemin du préfixe pour un nœud donné.
        
        Args:
            node: Nœud pour lequel trouver le chemin du préfixe
            
        Returns:
            Liste des chemins du préfixe avec leur support
        """
        prefix_paths = []
        
        while node is not None and node.parent is not None and node.parent.item is not None:
            prefix_paths.append(node.parent.item)
            node = node.parent
            
        return prefix_paths
    
    def _mine_tree(self, node, prefix, min_support_count):
        """
        Extrait récursivement les itemsets fréquents à partir du FP-Tree.
        
        Args:
            node: Nœud racine du FP-Tree
            prefix: Préfixe courant (itemset en cours de construction)
            min_support_count: Support minimal en nombre absolu
        """
        # Parcourir les items dans l'ordre inverse de fréquence pour
        # générer des arbres plus petits lors de l'extraction (optimisation)
        for item in reversed(self.ordered_items):
            new_prefix = prefix.copy()
            new_prefix.add(item)
            new_itemset = frozenset(new_prefix)
            
            # Ajouter cet itemset à la liste des itemsets fréquents s'il n'existe pas déjà
            support = self.header_table[item]['count']
            support_ratio = support / self.n_transactions
            
            self.frequent_itemsets_dict[new_itemset] = {
                'support_count': support,
                'support': support_ratio
            }
            
            # Construire la base de pattern conditionnelle
            conditional_pattern_base = []
            
            # Parcourir tous les nœuds contenant cet item
            node_ref = self.header_table[item]['head']
            while node_ref is not None:
                prefix_path = self._find_prefix_path(node_ref)
                if prefix_path:
                    conditional_pattern_base.append((prefix_path, node_ref.count))
                node_ref = node_ref.node_link
            
            # Construire le FP-Tree conditionnel si des chemins existent
            if conditional_pattern_base:
                conditional_tree = FPNode(None, 0)
                
                # Recalculer les supports dans cette base conditionnelle
                cond_item_counts = defaultdict(int)
                for path, count in conditional_pattern_base:
                    for path_item in path:
                        cond_item_counts[path_item] += count
                
                # Filtrer les items fréquents dans cette base conditionnelle
                frequent_items = {item: count for item, count in cond_item_counts.items() 
                                if count >= min_support_count}
                
                # Si des items fréquents existent, construire l'arbre conditionnel
                if frequent_items:
                    # Insérer les chemins dans l'arbre conditionnel
                    for path, count in conditional_pattern_base:
                        # Filtrer et trier les items du chemin
                        filtered_path = [p for p in path if p in frequent_items]
                        filtered_path.sort(key=lambda x: (-frequent_items[x], x))
                        
                        if filtered_path:
                            # Créer une mini table d'en-tête pour cet arbre conditionnel
                            header_table = {item: {'count': 0, 'head': None} for item in frequent_items}
                            self._insert_conditional_tree(filtered_path, conditional_tree, count, header_table)
                    
                    # Extraire récursivement les itemsets de cet arbre conditionnel
                    for cond_item in sorted(frequent_items, key=lambda x: (-frequent_items[x], x)):
                        newer_prefix = new_prefix.copy()
                        newer_prefix.add(cond_item)
                        newer_itemset = frozenset(newer_prefix)
                        
                        # Ajouter ce nouvel itemset fréquent
                        cond_support = frequent_items[cond_item]
                        cond_support_ratio = cond_support / self.n_transactions
                        
                        self.frequent_itemsets_dict[newer_itemset] = {
                            'support_count': cond_support,
                            'support': cond_support_ratio
                        }
        
    def _insert_conditional_tree(self, items, node, count, header_table):
        """
        Insère un chemin dans un arbre conditionnel.
        
        Args:
            items: Liste des items du chemin
            node: Nœud courant dans l'arbre
            count: Support du chemin
            header_table: Table d'en-tête pour cet arbre conditionnel
        """
        if not items:
            return
        
        item = items[0]
        
        # Mettre à jour le compteur dans la table d'en-tête
        header_table[item]['count'] += count
        
        # Insérer dans l'arbre
        if item in node.children:
            node.children[item].increment(count)
        else:
            new_node = FPNode(item, count, node)
            node.children[item] = new_node
            
            # Mettre à jour la table d'en-tête
            if header_table[item]['head'] is None:
                header_table[item]['head'] = new_node
            else:
                current = header_table[item]['head']
                while current.node_link:
                    current = current.node_link
                current.node_link = new_node
        
        # Insertion récursive
        self._insert_conditional_tree(items[1:], node.children[item], count, header_table)
    
    def _generate_rules(self):
        """
        Génère des règles d'association à partir des itemsets fréquents.

        Returns:
            Liste des règles d'association
        """
        self.rules_ = []
        
        # Examiner tous les itemsets de taille > 1
        for itemset, item_data in self.frequent_itemsets_dict.items():
            if len(itemset) < 2:
                continue
                
            # Pour chaque sous-ensemble possible comme antécédent
            for i in range(1, len(itemset)):
                for antecedent_items in self._subsets(itemset, i):
                    antecedent = frozenset(antecedent_items)
                    consequent = itemset - antecedent
                    
                    if antecedent in self.frequent_itemsets_dict and consequent in self.frequent_itemsets_dict:
                        # Calculer confiance et lift
                        confiance = item_data['support'] / self.frequent_itemsets_dict[antecedent]['support']
                        lift = confiance / self.frequent_itemsets_dict[consequent]['support']
                        
                        if confiance >= self.min_confiance:
                            self.rules_.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': item_data['support'],
                                'confiance': confiance,
                                'lift': lift
                            })
    
    def _subsets(self, s, n):
        """Génère tous les sous-ensembles de taille n d'un ensemble."""
        return [set(combo) for combo in combinations(s, n)]

    def get_frequent_itemsets(self):
        """
        Récupérer les itemsets fréquents découverts.
        
        Returns:
            List: Liste des itemsets fréquents avec leurs supports
        """
        # Convertir le dictionnaire en liste pour compatibilité
        result = [{'itemset': itemset, **item_data} for itemset, item_data in self.frequent_itemsets_dict.items()]
        return result

    def get_rules(self):
        """
        Récupérer les règles d'association générées.
        
        Returns:
            Liste des règles d'association
        """
        return self.rules_

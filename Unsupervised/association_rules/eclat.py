from itertools import combinations, chain
import time


class Eclat:
    """
    L'algorithme ECLAT est un algorithme de recherche en profondeur qui utilise une structure 
    de base de données verticale. Plutôt que de lister explicitement toutes les transactions, 
    chaque élément est associé à sa couverture (ou liste de transactions contenant cet élément). 
    L'approche par intersection est utilisée pour calculer le support des ensembles d'éléments. 
    Cet algorithme est particulièrement efficace pour les ensembles de données de petite taille 
    et nécessite moins d'espace et de temps que l'algorithme Apriori pour générer des motifs fréquents.

    Args:
        min_support (float): Seuil minimal de support (entre 0 et 1) pour les itemsets fréquents.
        min_confidence (float): Seuil minimal de confiance (entre 0 et 1) pour les règles d'association.
    """

    def __init__(self, min_support=0.1, min_confidence=0.7):
        if not 0 <= min_support <= 1:
            raise ValueError("Le support minimal doit être compris entre 0 et 1")
        if not 0 <= min_confidence <= 1:
            raise ValueError("La confiance minimale doit être comprise entre 0 et 1")

        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.frequent_TIDsets = {}
        self.rules_ = None
        self.n_transactions = 0

    def fit(self, transactions):
        """
        Méthode principale pour l'apprentissage des itemsets fréquents.
        
        Args:
            transactions: Liste de transactions (chaque transaction est un ensemble d'items)

        Returns:
            self: L'instance courante pour chaînage des méthodes
        """
        start_time = time.time()
        
        # Vérifier la conformité du format des données en entrée
        if isinstance(transactions, list):
            transactions_list = transactions
        else:
            raise TypeError("Format de données non respecté! List[set] format uniquement acceptés.")
        
        self.n_transactions = len(transactions_list)
        
        print(f"\nApplication de l'algorithme Eclat avec:")
        print(f"- Support minimum: {self.min_support} ({self.min_support*100}%)")
        print(f"- Confiance minimum: {self.min_confidence} ({self.min_confidence*100}%)")

        print(f"Nombre de transactions valides : {self.n_transactions}")

        # Trouver les itemsets fréquents
        self._fit_eclat(transactions_list)

        # Générer les règles d'association
        self._generate_rules()

        elapsed_time = time.time() - start_time

        print(f"\nTemps d'exécution: {elapsed_time:.2f} secondes")
        return self

    def _fit_eclat(self, transactions):
        """
        Implémentation de la phase d'extraction des itemsets fréquents.
        
        Args:
            transactions: Liste de transactions (chaque transaction est un ensemble d'items)
        """
        # Récupérer tous les items uniques dans les transactions
        all_items = set(chain(*transactions))
        
        # Stocker les items fréquents de taille 1
        self.frequent_itemsets[1] = self.get_single_items_TIDset(all_items, transactions)
        
        # Générer des itemsets de taille croissante
        k = 1
        while self.frequent_itemsets.get(k):            
            k += 1
            candidates = self._generate_candidates(k)
            
            if not candidates:
                break
                
            # Calculer les TIDsets pour les nouveaux candidats
            level_frequent = {}
            for candidate in candidates:
                subsets = [frozenset(subset) for subset in combinations(candidate, k-1)]
                if all(subset in self.frequent_itemsets[k-1] for subset in subsets):
                    # Intersection des TIDsets des sous-ensembles
                    TIDset = set.intersection(*[self.frequent_itemsets[k-1][subset] for subset in subsets])
                    
                    # Vérifier le support
                    if len(TIDset) / self.n_transactions >= self.min_support:
                        level_frequent[candidate] = TIDset
            
            if level_frequent:
                self.frequent_itemsets[k] = level_frequent
            else:
                break

    def get_single_items_TIDset(self, items, transactions):
        """Construction des TIDsets pour les items individuels"""
        single_items_TIDsets = {}
        for item in items:
            TIDset = set()
            for tid, transaction in enumerate(transactions):
                if item in transaction:
                    TIDset.add(tid)
            
            # Vérifier si l'item atteint le support minimum
            if len(TIDset) / self.n_transactions >= self.min_support:
                single_items_TIDsets[frozenset([item])] = TIDset

        return single_items_TIDsets
    
    def _generate_candidates(self, k):
        """
        Générer des candidats de taille k à partir des itemsets fréquents de taille k-1.
        
        Args:
            k: Taille des candidats à générer
            
        Returns:
            Ensemble des candidats générés
        """
        candidates = set()
        prev_frequent = self.frequent_itemsets.get(k-1, {})
        
        for itemset1, itemset2 in combinations(prev_frequent.keys(), 2):
            # Jointure d'itemsets ayant k-2 éléments en commun
            union = itemset1.union(itemset2)
            if len(union) == k:
                # Vérifier que tous les sous-ensembles de taille k-1 sont fréquents
                if all(frozenset(subset) in prev_frequent for subset in combinations(union, k-1)):
                    candidates.add(union)
        
        return candidates

    def _generate_rules(self):
        """
        Générer des règles d'association à partir des itemsets fréquents.
        """
        self.rules_ = []
        
        # Parcourir tous les itemsets fréquents de taille > 1
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
                
            for itemset, TIDset in self.frequent_itemsets[k].items():
                itemset_support = len(TIDset) / self.n_transactions
                
                # Générer toutes les règles possibles à partir de cet itemset
                for i in range(1, k):
                    for antecedent_items in combinations(itemset, i):
                        antecedent = frozenset(antecedent_items)
                        consequent = itemset.difference(antecedent)
                        
                        # Calculer la confiance
                        antecedent_support = len(self.frequent_itemsets[len(antecedent)][antecedent]) / self.n_transactions
                        confidence = itemset_support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            # Calculer le lift
                            consequent_support = len(self.frequent_itemsets[len(consequent)][consequent]) / self.n_transactions
                            lift = confidence / consequent_support
                            
                            # Ajouter la règle à la liste
                            self.rules_.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': itemset_support,
                                'confidence': confidence,
                                'lift': lift
                            })


    def get_frequent_itemsets(self):
        """
        Récupérer les itemsets fréquents découverts.
        
        Returns:
            dict: Dictionnaire des itemsets fréquents où les clés sont les tailles
                et les valeurs sont des ensembles d'itemsets
        """
        return self.frequent_itemsets
    
    def get_rules(self):
        """
        Accesseur pour récupérer les règles d'association générées.
        
        Returns:
            Liste des règles d'association
        """
        return self.rules_

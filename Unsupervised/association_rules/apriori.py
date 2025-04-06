from utils import DataAdapter
from metrics import support, confidence, lift
from itertools import chain, combinations
from collections import defaultdict
import time

class Apriori:
    """
    L'algorithme Apriori est utilisé pour découvrir des règles d'association 
    intéressantes dans de grands ensembles de données transactionnels.

    Args:
        min_support(float): Support minimal pour considérer un itemset
        min_confiance(float): Seuil minimal de confiance pour une règle
    """
    def __init__(self, min_support: float, min_confiance: float):
        
        if not 0 <= min_support <= 1:
            raise ValueError("Le support minimal doit être compris entre 0 et 1")
        if not 0 <= min_confiance <= 1:
            raise ValueError("La confiance minimale doit être comprise entre 0 et 1")
        
        self.min_support = min_support
        self.min_confiance = min_confiance
        self.frequent_itemsets_ = None
        self.rules_ = None

    def fit(self, transactions: list):
        """
        Méthode principale pour l'apprentissage des itemsets fréquents et des règles.
        
        Args:
            transactions: Données d'entrée (list[set])
            
        Returns:
            self: L'instance courante pour chaînage des méthodes
        """
        start_time = time.time()
        
        # Vérifier la conformité du format des données en entrée
        if isinstance(transactions, list):
            transactions_list = DataAdapter._convert_from_list(transactions)
        else:
            raise TypeError("Format de données non respecté! List[set] format uniquement acceptés.")
        
        print(f"\nApplication de l'algorithme Apriori avec:")
        print(f"- Support minimum: {self.min_support} ({self.min_support*100}%)")
        print(f"- Confiance minimum: {self.min_confiance} ({self.min_confiance*100}%)")

        print(f"Nombre de transactions valides : {len(transactions_list)}")
        if transactions_list:
            print(f"Exemple de transaction : {list(transactions_list[0])[:5]}...")

        # Générer les itemsets fréquents
        self._fit_apriori(transactions_list)
        # Générer les règles d'association
        self._generate_rules(transactions_list)

        elapsed_time = time.time() - start_time

        print(f"\nTemps d'exécution: {elapsed_time:.2f} secondes")
        return self
    
    def _fit_apriori(self, transactions: list[set]):
        """
        Extraction des k-itemsets fréquents

        ## Étapes:
            1. Extraire tous les items uniques
            2. Trouver les 1-itemsets fréquents
            3. Générer itérativement des itemsets de taille croissante
        """
        # Récupérer les items uniques
        items = set(chain(*transactions))
        # Récupérer les 1-itemsets fréquents
        self.frequent_itemsets_ = {
            1: self._get_one_itemsets(items, transactions)
        }
        
        # Générer les itemsets de taille croissante
        size = 1
        while True:
            size += 1
            candidates = self._generate_candidates(self.frequent_itemsets_[size-1])
            frequent = self._prune_candidates(candidates, transactions)

            # Arrêter si aucun item fréquent n'est trouvé
            if not frequent: 
                break
            # Stocker les itemsets fréquents de cette taille
            self.frequent_itemsets_[size] = frequent

    def _get_one_itemsets(self, items: set, transactions: list[set]):
        """
        Calcule les 1-itemsets fréquents (items individuels).
        
        Args:
            items: Ensemble de tous les items uniques
            transactions: Liste des transactions
        
        Returns:
            Ensemble des items fréquents respectant le support minimal
        """
        # Occurence de chaque item unique dans la base de données
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
        Génère de nouveaux candidats en combinant les itemsets précédents.
        Utilise l'approche optimisée "join-and-prune" qui combine uniquement 
        les itemsets partageant les mêmes k-1 premiers éléments.
        
        Args:
            previous_itemsets: Itemsets de la taille précédente k
        
        Returns:
            Nouveaux candidats de taille supérieure k + 1
        """
        candidates = set()
        previous_list = list(previous_itemsets)
        k = len(list(previous_list[0])) if previous_list else 0
        
        # Phase de jointure
        for i in range(len(previous_list)):
            for j in range(i+1, len(previous_list)):
                # Convertir en liste pour pouvoir comparer les éléments par index
                items1 = sorted(list(previous_list[i]))
                items2 = sorted(list(previous_list[j]))
                
                # Vérifier si les k-1 premiers éléments sont identiques
                if items1[:k-1] == items2[:k-1]:
                    # Créer un nouvel itemset candidat
                    new_candidate = frozenset(previous_list[i] | previous_list[j])
                    
                    # Phase d'élagage - tous les sous-ensembles doivent être fréquents
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
        Filtre les candidats selon le support minimal de manière optimisée.
        
        Args:
            candidates: Ensemble des candidats à tester
            transactions: Liste des transactions
        
        Returns:
            Itemsets fréquents parmi les candidats
        """
        if not candidates:
            return set()
            
        return {c for c in candidates if support(c, transactions) >= self.min_support}
    
    def _generate_rules(self, transactions: list[set]):
        """
        Génère les règles d'association à partir des itemsets fréquents.
        Calcule la confiance et le lift pour chaque règle.
        
        Args:
            transactions: Liste des transactions
        """
        self.rules_ = []
        
        # Parcourir les itemsets fréquents en ignorant ceux de taille 1
        for itemset in chain(*(self.frequent_itemsets_.values())):
            if len(itemset) < 2:
                continue

            # Générer toutes les combinaisons possibles de règles
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    # Calculer les métriques
                    conf = confidence(antecedent, consequent, transactions)
                    
                    if conf >= self.min_confiance:
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
        Récupérer les itemsets fréquents découverts.
        
        Returns:
            dict: Dictionnaire des itemsets fréquents où les clés sont les tailles
                et les valeurs sont des ensembles d'itemsets
        """
        return self.frequent_itemsets_

    def get_rules(self):
        """
        Récupérer les règles d'association générées.
        
        Returns:
            Liste des règles d'association
        """
        return self.rules_
    

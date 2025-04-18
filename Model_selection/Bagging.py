import numpy as np 
from utils import clone

class BaggingRegressor:
    ''' 
        BaggingRegressor est une implémentation du bagging (Bootstrap Aggregating) pour des modèles de régression.
        Cette méthode agrège les prédictions des modèles de régression en calculant la moyenne 
        des prédictions de plusieurs sous-modèles entraînés sur des échantillons bootstrap(avec remise) du jeu de données original.

        Attributs :

        base_model : Le modele de regression de base à utiliser pour le bagging
        
        n_estimators (int) : Le nombre de modèles à entraîner. Par défaut, la valeur est 10.
        
        random_state : pour la reproductibilité. Default = None
        
        Retourne:
        La moyenne des predictions des modeles
        '''
    def __init__(self, base_model, n_estimators=10, random_state=None):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        ''' 
        La méthode fit entraîne les différentes versions du modele apres echantillonage boostrap
        Attributs :
        - X : les features
        - y : la target

        '''
        X, y = np.array(X), np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X et y doivent avoir le même nombre d'échantillons")
            
        if not (hasattr(self.base_model, 'fit') and hasattr(self.base_model, 'predict')):
            raise ValueError("Le modèle de base doit implémenter fit() et predict()")
        
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Échantillonner avec remise
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            #model = self.base_model
            model = clone(self.base_model)            
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        X = np.array(X)
        predictions = np.zeros((self.n_estimators, X.shape[0]))
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X)

        return np.mean(predictions, axis=0)

class BaggingClassifier:
    ''' 
        BaggingClassifier est une implémentation du bagging (Bootstrap Aggregating) pour des modèles de classification.
        Cette méthode agrège les prédictions des modèles de classification en faisant un vote majoritaire (la classe ayant le plus été choisi par les modeles sera 
        finalement attribuée à la donnée à classer) des prédictions de plusieurs sous-modèles entraînés sur des échantillons bootstrap(avec remise) du jeu de données original.

        Attributs :

        base_model : Le modele de regression base à utiliser pour le bagging
        
        n_estimators (int) : Le nombre de modèles à entraîner. Par défaut, la valeur est 10.
        
        random_state : pour la reproductibilité. Default = None

        Retourne:
        Les classes predites apres vote majoritaire
    '''

    def __init__(self, base_model, n_estimators=10, random_state = None):
        
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X et y doivent avoir le même nombre d'échantillons")
            
        if not (hasattr(self.base_model, 'fit') and hasattr(self.base_model, 'predict')):
            raise ValueError("Le modèle de base doit implémenter fit() et predict()")
        
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Échantillonner avec remise
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            model = clone(self.base_model)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros((self.n_estimators, X.shape[0]))
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X)

        return np.array([np.argmax(np.bincount(predictions[:, i].astype(int))) for i in range(X.shape[0])])

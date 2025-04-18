import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from cross_validation import k_fold_cross_validation

class GaussianProcess:
    """
    processus gaussien avec noyau RBF (Radial Basis Function).
    Permet de modéliser la fonction objectif à optimiser.

    Attributs :
        kernel_var : Variance du noyau RBF.
        length_scale : Échelle de longueur du noyau.
        noise : Bruit ajouté au noyau (régularisation).
    """

    def __init__(self, kernel_var=1.0, length_scale=1.0, noise=1e-6):
        self.kernel_var = kernel_var
        self.length_scale = length_scale
        self.noise = noise

    def rbf_kernel(self, X1, X2):
        """matrice de covariance (noyau RBF) entre X1, X2"""
        dists = cdist(X1, X2, 'sqeuclidean')  # distance euclidienne au carré
        return self.kernel_var * np.exp(-0.5 * dists / self.length_scale**2)

    def fit(self, X_train, y_train):
        """ apprentissage modèle GPR en ajustant la matrice de covariance. """
        self.X_train = X_train
        self.y_train = y_train
        K = self.rbf_kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test):
        """ prédiction moyenne, écart-type pour nouveaux points. """
        K_s = self.rbf_kernel(self.X_train, X_test)
        K_ss = self.rbf_kernel(X_test, X_test) + self.noise * np.eye(len(X_test))
        mu = K_s.T @ self.K_inv @ self.y_train
        cov = K_ss - K_s.T @ self.K_inv @ K_s
        return mu, np.sqrt(np.diag(cov))


def expected_improvement(X, gp, y_min, xi=0.01):
    """
    Expected Improvement (EI) pour les points X donnés.

    Arguments :
        X : échantillons candidats (2D)
        gp : modèle GaussianProcess entraîné
        y_min : meilleure performance observée jusque-là
        xi : facteur d’exploration (plus grand = plus d’exploration)

    Retour :
        ei : valeur de l'Expected Improvement pour chaque point.
    """
    mu, sigma = gp.predict(X)
    with np.errstate(divide='warn'):
        imp = y_min - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0  # éviter division par zéro
    return ei



class BayesianSearchCV:
    """
    Optimisation bayésienne avec validation croisée pour la recherche d'hyperparamètres.

    Arguments :
        estimator : modèle sklearn-like avec .fit() et .predict()
        param_bounds : dictionnaire des bornes des hyperparamètres
        n_iter : nombre d'itérations de recherche
        init_points : nombre d’échantillons aléatoires pour initialisation
        cv : nombre de folds pour la validation croisée
        scoring : fonction de scoring (doit avoir attribut _score_func)
    """

    def __init__(self, estimator, param_bounds,scoring, stratified= None, n_iter=20, init_points=5, cv=5):
        self.estimator = estimator
        self.param_bounds = param_bounds
        self.n_iter = n_iter
        self.init_points = init_points
        self.cv = cv
        self.scoring = scoring
        self.stratified = stratified
        self.X_obs = []  # liste vecteurs d’hyperparamètres testés
        self.y_obs = []  # scores correspondants
        self.gp = GaussianProcess()

    def _sample_params(self):
        """vecteur d’hyperparamètres aléatoires dans les bornes spécifiées"""
        return np.array([np.random.uniform(low, high) for (low, high) in self.param_bounds.values()])

    def _dict_from_vector(self, x_vector):
        """ transformation vecteur en dictionnaire de paramètres. """
        return dict(zip(self.param_bounds.keys(), x_vector))


    def _evaluate(self, X, y, x):
        """ Évalue un jeu de paramètres (x) sur les données (X, y). """
        if not isinstance(x, dict):
            x = self._dict_from_vector(x)

        # Exemple : forcer 'n_neighbors' à être entier
        if "n_neighbors" in x:
            x["n_neighbors"] = int(round(x["n_neighbors"]))
            if x["n_neighbors"] < 1:
                raise ValueError("n_neighbors doit être un entier positif")

        # appliquer paramètres au modèle
        self.estimator.set_params(**x)
        mean_score, _ = k_fold_cross_validation(self.estimator, X, y, self.scoring, self.stratified, k=self.cv)
        return mean_score

    def _suggest(self, n_candidates=100):
        """ propose nouvel échantillon via Expected Improvement. """
        X_candidates = np.array([self._sample_params() for _ in range(n_candidates)])
        ei = expected_improvement(X_candidates, self.gp, y_min=np.max(self.y_obs))
        return X_candidates[np.argmax(ei)]

    def fit(self, X, y):
        """
        Exécute la recherche bayésienne sur les données (X, y).
        Enregistre les meilleurs paramètres dans .best_params_ et le meilleur score dans .best_score_.
        """
        X, y = np.array(X), np.array(y)

        # Phase initiale : points aléatoires
        for _ in range(self.init_points):
            x = self._sample_params()
            y_score = self._evaluate(X, y, x)
            self.X_obs.append(x)
            self.y_obs.append(y_score)

        # Boucle d'optimisation
        for i in range(self.n_iter):
            self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))
            x_next = self._suggest()
            y_next = self._evaluate(X, y, x_next)
            self.X_obs.append(x_next)
            self.y_obs.append(y_next)
            print(f"[{i+1}/{self.n_iter}] Score = {y_next:.4f}")

        # récupérer meilleur jeu de paramètres
        best_idx = np.argmin(self.y_obs)  # minimisation si le score est une erreur
        self.best_params_ = self._dict_from_vector(self.X_obs[best_idx])
        self.best_score_ = self.y_obs[best_idx]

    def get_best_params(self):
        """ retourner meilleurs paramètres trouvés. """
        return self.best_params_

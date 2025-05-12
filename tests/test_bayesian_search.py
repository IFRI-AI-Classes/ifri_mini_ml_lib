import numpy as np
from ifri_mini_ml_lib.model_selection.utils import BaseEstimatorMinimal
from ifri_mini_ml_lib.metrics.evaluation_metrics import accuracy
from ifri_mini_ml_lib.model_selection.bayesian_searchCV import BayesianSearchCV, GaussianProcess, expected_improvement

class KNeighborsClassifierSimple(BaseEstimatorMinimal):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculer la distance Euclidienne et trouver les k voisins
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_neighbors = self.y_train[k_indices]
            # Prédire la classe majoritaire parmi les k voisins
            predictions.append(np.bincount(k_neighbors).argmax())
        return np.array(predictions)

class LinearRegressionSimple(BaseEstimatorMinimal):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # Ajouter un biais (colonne de 1) aux données
        X_ = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.linalg.inv(X_.T @ X_) @ X_.T @ y
    
    def predict(self, X):
        X_ = np.c_[np.ones(X.shape[0]), X]
        return X_ @ self.coef_
      
def testbay():
    #Quelques données fictives
    X = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2]])
    y = np.array([0, 0, 1])
    
    model = KNeighborsClassifierSimple()
    
    param_bounds={'n_neighbors': (1, 20)}
    
    param_types={'n_neighbors': 'int'}
    
    search = BayesianSearchCV(
    estimator=KNeighborsClassifierSimple(),
    param_bounds= param_bounds,
    param_types = param_types,
    n_iter=5,
    init_points=3,
    cv=3,
    scoring=accuracy,
    maximize=True
    )
    
    search.fit(X,y)
    
    best_params = search.get_best_params()
    assert isinstance (best_params, dict)
    assert 'n_neighbors' in best_params
    assert 1 <= best_params['n_neighbors']<=20
 
def testgp():
    X_train = np.array([[1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 2.0, 3.0])
    
    X_test = np.array([[1.5], [2.5]])
    
    gp = GaussianProcess(kernel_var=1.0, length_scale=1.0, noise=1e-6)
    
    gp.fit(X_train, y_train)
    
    mu, sigma = gp.predict(X_test)
    
    assert mu.shape == (2,)
    assert sigma.shape == (2,)
    assert np.all(sigma >= 0)
    
def testei():
    X_train = np.array([[0.0], [1.0]])
    y_train = np.array([0.5, 0.2])
    X_candidates = np.array([[0.5], [1.5]])
    gp = GaussianProcess()
    gp.fit(X_train, y_train)
    ei = expected_improvement(X_candidates, gp, y_min=np.min(y_train))
    
    assert ei.shape == (2,)
    assert np.all(ei >= 0) 

def mean_squared_error(y_true, y_pred):
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
   
def testbayreg():
    
    np.random.seed(0)
    X = np.random.rand(100, 3)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(100) * 0.1

    
    model = LinearRegressionSimple()

    
    param_bounds = {'dummy_param': (0.0, 1.0)}
    param_types = {'dummy_param': 'float'}

    
    search = BayesianSearchCV(
        estimator=model,
        param_bounds=param_bounds,
        param_types=param_types,
        scoring=lambda y_true, y_pred: -mean_squared_error(y_true, y_pred),
        n_iter=3,
        init_points=2,
        cv=3,
        maximize=False
    )

    
    def set_params_with_dummy(self, **params):
        return self
    model.set_params = set_params_with_dummy.__get__(model, LinearRegressionSimple)

    
    search.fit(X, y)
    best = search.get_best_params()

    
    assert 'dummy_param' in best
    assert 0.0 <= best['dummy_param'] <= 1.0  
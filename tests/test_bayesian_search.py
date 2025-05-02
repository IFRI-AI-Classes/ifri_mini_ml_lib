import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from ifri_mini_lib.model_selection import BayesianSearchCV
from ifri_mini_lib.model_selection.bayesian_searchCV import GaussianProcess, expected_improvement

def testbay():
    X, y = load_iris(return_X_y=True)
    
    model = KNeighborsClassifier()
    
    param_bounds={'n_neighbors': (1, 20)}
    
    param_types={'n_neighbors': 'int'}
    
    search = BayesianSearchCV(
    estimator=KNeighborsClassifier(),
    param_bounds= param_bounds,
    param_types = param_types,
    n_iter=5,
    init_points=3,
    cv=3,
    scoring=accuracy_score
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
    
def testbayreg():
    
    np.random.seed(0)
    X = np.random.rand(100, 3)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(100) * 0.1

    
    model = LinearRegression()

    
    param_bounds = {'dummy_param': (0.0, 1.0)}
    param_types = {'dummy_param': 'float'}

    
    search = BayesianSearchCV(
        estimator=model,
        param_bounds=param_bounds,
        param_types=param_types,
        scoring=lambda y_true, y_pred: -mean_squared_error(y_true, y_pred),
        n_iter=3,
        init_points=2,
        cv=3
    )

    
    def set_params_with_dummy(self, **params):
        return self
    model.set_params = set_params_with_dummy.__get__(model, LinearRegression)

    
    search.fit(X, y)
    best = search.get_best_params()

    
    assert 'dummy_param' in best
    assert 0.0 <= best['dummy_param'] <= 1.0  
from bayesian_searchCV import BayesianSearchCV
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from time import time

# Charger les données
data = load_iris()
X, y = data.data, data.target

# Définir le scoring
scoring = accuracy_score


my_search = BayesianSearchCV(
    estimator=KNeighborsClassifier(),
    param_bounds={'n_neighbors': (1, 20)},
    param_types={'n_neighbors': 'int'},
    n_iter=10,
    init_points=3,
    cv=3,
    scoring=scoring
)

start = time()
my_search.fit(X, y)
my_duration = time() - start

print("\n[Pour nous]")
print("Best params:", my_search.get_best_params())
print("Best score:", my_search.best_score_)
print("Duration: %.2fs" % my_duration)


opt_search = BayesSearchCV(
    KNeighborsClassifier(),
    search_spaces={'n_neighbors': (1, 20)},
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=0
)

start = time()
opt_search.fit(X, y)
opt_duration = time() - start

print("\n[Por skopt]")
print("Best params:", opt_search.best_params_)
print("Best score:", opt_search.best_score_)
print("Duration: %.2fs" % opt_duration)

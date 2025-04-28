from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from random_searchCV import RandomSearchCV_Scratch
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Définir les distributions de paramètres pour Random Search
param_dist = {
    'n_neighbors': [1, 3, 7, 9, 11, 13, 15, 17, 19, 21],  # Entiers entre 3 et 9
    'weights': ['uniform', 'distance'],
}

# RandomSearchCV_Scratch
random_search_scratch = RandomSearchCV_Scratch(
    KNeighborsClassifier(), 
    param_dist,
    scoring= accuracy_score,
    n_iter=20,
    stratified= False,
)
random_search_scratch.fit(X, y)

# RandomizedSearchCV de scikit-learn
random_search_cv = RandomizedSearchCV(
    KNeighborsClassifier(), 
    param_dist,
    n_iter=20,
    cv=5,
)
random_search_cv.fit(X, y)

# Afficher les meilleurs paramètres et scores
print("\n=== Résultats sklearn ===")
print("RandomizedSearchCV - Meilleurs paramètres:", random_search_cv.best_params_)
print("RandomizedSearchCV - Meilleur score:", random_search_cv.best_score_)

print("\n=== Résultats from scratch ===")
print("RandomSearchCV_Scratch - Meilleurs paramètres:", random_search_scratch.best_params_)
print("RandomSearchCV_Scratch - Meilleur score:", random_search_scratch.best_score_)


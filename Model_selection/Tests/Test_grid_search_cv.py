import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_iris
from sklearn.neighbors import KNeighborsClassifier
from grid_searchCV import GridSearchCV_Scratch
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

'''data = load_iris()
X = data.data
y = data.target'''

X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Définir les hyperparamètres à tester
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9 ],  
    'weights': ['uniform', 'distance']
}

# GridSearchCV_Scratch
grid_search_scratch = GridSearchCV_Scratch(KNeighborsClassifier(), param_grid, scoring=accuracy_score, stratified=True)
grid_search_scratch.fit(X, y)

# GridSearch de scikit-learn
grid_search_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search_cv.fit(X, y)

# Afficher les meilleurs paramètres et scores
print("\n=== Résultats sklearn ===")
print("GridSearchCV - Meilleurs paramètres: ", grid_search_cv.best_params_)
print("GridSearchCV - Meilleur score: ", grid_search_cv.best_score_)

print("\n=== Résultats from scratch ===")
print("GridSearchCV_Scratch - Meilleurs paramètres : ", grid_search_scratch.best_params_)
print("GridSearchCV_Scratch - Meilleur score : ", grid_search_scratch.best_score_)



'''df = pd.read_csv("sonar data.csv", header=None)
label_mapping = {'M': 1, 'R': 0}
df.iloc[:, -1] = df.iloc[:, -1].map(label_mapping).astype(int)

X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values 
y = y.astype(int)

param_grid = {
    "n_neighbors": [16, 9, 11, 13, 15]
}



grid_search_scratch = GridSearchCV_Scratch(KNeighborsClassifier(), param_grid)
grid_search_scratch.fit(X, y)

grid_search_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search_cv.fit(X, y)

# Afficher les meilleurs paramètres et scores
print("\n=== Résultats sklearn ===")
print("GridSearchCV - Meilleurs paramètres: ", grid_search_cv.best_params_)
print("GridSearchCV - Meilleur score: ", grid_search_cv.best_score_)

print("\n=== Résultats from scratch ===")
print("GridSearchCV_Scratch - Meilleurs paramètres : ", grid_search_scratch.best_params_)
print("GridSearchCV_Scratch - Meilleur score : ", grid_search_scratch.best_score_)'''

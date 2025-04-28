from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import clone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier as SklearnBaggingClassifier
import numpy as np
from bagging import BaggingClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#Entrainons nos propres modeles
n_estimators = 3
base_model = DecisionTreeClassifier(random_state=42)
'''models = []

for _ in range(n_estimators):
    # Échantillon bootstrap
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    model = clone(base_model)
    model.fit(X_train[indices], y_train[indices])
    models.append(model)'''

# Entraînons BaggingClassifier from scratch
#Case 1: base model
bagging_from_scratch = BaggingClassifier(base_model, n_estimators=10)
bagging_from_scratch.fit(X_train, y_train)

#Case 2 pretrained models
#bagging_from_scratch = BaggingClassifier(pretrained_models=models)
#bagging_from_scratch.fit(X_train, y_train)

# Prédictions sur le jeu de test
y_pred_from_scratch = bagging_from_scratch.predict(X_test)

accuracy_from_scratch = accuracy_score(y_test, y_pred_from_scratch)
print(f'Accuracy de BaggingClassifier from scratch : {accuracy_from_scratch}')

# Comparons avec le BaggingClassifier de sklearn
sklearn_bagging = SklearnBaggingClassifier(base_model, n_estimators=3, random_state=42)
sklearn_bagging.fit(X_train, y_train)
y_pred_sklearn = sklearn_bagging.predict(X_test)

# Accuracy de sklearn
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f'Accuracy de BaggingClassifier de sklearn : {accuracy_sklearn}')

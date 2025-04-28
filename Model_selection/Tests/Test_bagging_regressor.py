from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor as SklearnBaggingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from bagging import *

# Charger les données
import pandas as pd


data_url = "http://lib.stat.cmu.edu/datasets/boston"     
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Notre modèle BaggingRegressor from scratch
bagging_regressor = BaggingRegressor(base_model=DecisionTreeRegressor(), n_estimators=100)
bagging_regressor.fit(X_train, y_train)
predictions_from_scratch = bagging_regressor.predict(X_test)

# Comparaison avec Sklearn BaggingRegressor
sklearn_bagging_regressor = SklearnBaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=100)
sklearn_bagging_regressor.fit(X_train, y_train)
predictions_from_sklearn = sklearn_bagging_regressor.predict(X_test)

# Calculer les MSE pour comparer
mse_from_scratch = mean_squared_error(y_test, predictions_from_scratch)
mse_from_sklearn = mean_squared_error(y_test, predictions_from_sklearn)

print(f"MSE BaggingRegressor from scratch: {mse_from_scratch}")
print(f"MSE BaggingRegressor from sklearn: {mse_from_sklearn}")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Regressions import LinearRegression
from metrics import evaluate_model

# Génération de données factices
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 3 * X + 5 + np.random.randn(100)
df = pd.DataFrame({'X': X, 'y': y})

# Création et entraînement du modèle
model = LinearRegression(df=df, method="gradient_descent")
model.split_data(test_size=0.2)
model.fit()

# Prédictions
y_pred = model.predict(model.test_X)

# Évaluation
metrics = evaluate_model(model.test_y, y_pred)

# Affichage des métriques
print("=== Évaluation du modèle ===")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Visualisation des prédictions vs valeurs réelles
plt.scatter(range(len(model.test_y)), model.test_y, label='Vrai')
plt.scatter(range(len(y_pred)), y_pred, label='Prédit', marker='x')
plt.legend()
plt.title("Prédictions vs Réel")
plt.xlabel("Index")
plt.ylabel("Valeur")
plt.grid(True)
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

#from perceptron import Perceptron
from perceptron_classifier import PerceptronClassifier
from perceptron_regressor import PerceptronRegressor


# EXAMPLES WITH SOME CONTINUOUS DATA
# DATA - `LOGISTIC REG_`|
X_REG0 = [[87, 81],
      [79, 75],
      [65, 64],
      [53, 58],
      [41, 49]]
# y = E_part( div(x,10) )
Y_REG0 = [8, 7, 6, 5, 4]

# EXAMPLES WITH SOME LOGIC GATES [INPUT]
# DATA - `BINARY CLASS_`|
#         x1  x2        |
X_BIN = [ [0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]

Y_AND = [0, 0, 0, 1]  # y
Y_OR =  [0, 1, 1, 1]  # y
Y_XOR = [0, 1, 1, 0]  # y

# MODELS     |
#pct = Perceptron()
pctLR = PerceptronRegressor(n_iter=21)
pctBC = PerceptronClassifier(n_iter=100)

# TRAINING   |
# PREDICTION |
# ------------
pctLR.fit(X=X_REG0, y=Y_REG0) # 0
p0 = pctLR.predict(X=X_REG0)
print(f"Y_REG0_pred : {p0}")

pctBC.fit(X=X_BIN, y=Y_AND) # 1 : AND
p1 = pctBC.predict(X=X_BIN)
print(f"Y_AND_pred : {p1}")

pctBC.fit(X=X_BIN, y=Y_OR) # 2 : OR
p2 = pctBC.predict(X=X_BIN)
print(f"Y_OR_pred : {p2}")

pctBC.fit(X=X_BIN, y=Y_XOR) # 3 : XOR
p3 = pctBC.predict(X=X_BIN)
print(f"Y_XOR_pred : {p3}")

# OUTPUT examples |
# -----------------
# Y_REG0_pred : [1.65374203e+306 1.51726659e+306 1.27305831e+306 1.10060269e+306 8.95889525e+305]
# Y_AND_pred : [0 0 0 1]
# Y_OR_pred : [0 1 1 1]
# Y_XOR_pred : [1 1 0 0]

# RAPPORT         |
# Y_REG0_pred : Exponential results cuz __class__ skipped the 'stopping criteria'
# NotImplemented method

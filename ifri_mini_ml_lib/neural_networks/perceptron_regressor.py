import numpy as np


class PerceptronRegressor:
  """
  **Perceptron for *`logistic regression`***
  
  Parameters
  -----------
    learning_rate : float
      The learning rate ( 𝜂 ) controls the size of the steps that the model takes when it adjusts its weights.
    n_iter : int
      maximum iteration
  
  Attributes
  ----------
    weights : np.ndarray
      w ∈ R^N.
      w = {w1, w2, ..., wN} each corresponding to an input and indicating its importance.
    bias : float
      bias ∈ R.
      Constant term that allows shifting of the activation function to control the threshold for firing.
  
  Examples
  --------
  Create an `Self@PerceptronRegressor` object
  
  ```
  from perceptron_regressor import PerceptronRegressor
  pctLR = PerceptronRegressor(learning_rate=0.1, n_iter=1000)
  ```

  Main formulas
  -------------
  
  .. math::
  ```
  activation_function <- z = w^T . x + b
  error <- y_err = y - yp
  weights <- w = w + η(y - yp) . x
  bias <- b = b + η(y - yp)
  ```
  
  References
  ----------
  - https://www.geeksforgeeks.org/deep-learning/what-is-perceptron-the-simplest-artificial-neural-network/

  """
  
  def __init__(self, learning_rate=0.1, n_iter=1000):
    """
    Create an `Self@PerceptronRegressor` object
    """
    self.lr = learning_rate
    self.n_iter = n_iter
    self.weights : np.ndarray
    self.bias : float # b ∈ R

  def fit(self, X:list|np.ndarray, y:list|np.ndarray):
    # convert if dtypes='list'
    X, y = np.array(X, dtype=float), np.array(y, dtype=float)

    # initialize weights and bias
    self.weights = np.zeros(shape=X.shape[1])
    self.bias = 0

    # updates rules come from `Gradient Descent` Algorithm
    for _ in range(self.n_iter):
      for i, Xi in enumerate(X):
        y_pred = np.dot(Xi, self.weights) + self.bias # linear activation function
        y_err = y[i] - y_pred # error
        # update parameters (GD)
        self.weights += self.lr * y_err * Xi
        self.bias += self.lr * y_err
      
  def predict(self, X:list|np.ndarray):
    yGpred = np.dot(X, self.weights) + self.bias
    return yGpred


import numpy as np

from .data_split import split_train_validation


class Perceptron:
  """
  **Perceptron class for *`binary classification`***
  
  History
  -------
  - 1958: Frank Rosenblatt introduced the perceptron as a binary classifier.
  - 1969: Minsky and Papert's book "Perceptrons" highlighted limitations of single-layer perceptrons, leading to a decline in interest.
  - 1980s: The backpropagation algorithm enabled training of multi-layer perceptrons, revitalizing interest in neural networks.
  - 2010s: Advances in computing power and data availability led to the resurgence of deep learning, with multi-layer perceptrons (MLPs) becoming a fundamental building block for more complex architectures.
  - 2020s: Perceptrons and MLPs continue to be widely used for various applications, including image recognition, natural language processing, and more.  

  Source: https://en.wikipedia.org/wiki/Perceptron
  
  Parameters
  -----------
    lr : float
      The learning rate ( 𝜂 ) controls the size of the steps that the model takes when it adjusts its weights.
    n_iter : int
      Maximum iteration.
    early_stopping : bool, default=True
        
    patience : int, default=10
        
    validation_split : float, default=0.2
        
    shuffle : bool, default=True
        
    random_state : int, default=None
        
  
  Attributes
  ----------
    weights : np.ndarray
      w ∈ R^N.
      w = {w1, w2, ..., wN} each corresponding to an input and indicating its importance.
    bias : float
      bias ∈ R.
      Constant term that allows shifting of the activation function to control the threshold for firing.
    trained_epochs : int
      ...
  
  Examples
  --------

  Create, train and test an `Perceptron` model with various data
  ```
  import numpy as np
  from ifri_mini_ml_lib.neural_networks.perceptron import Perceptron
  
  # EXAMPLES WITH SOME LOGIC GATES [INPUT]
  # DATA - `BINARY CLASS_`|
  #         x1  x2        |
  X_BIN = [ [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]]
  X_BIN_V = np.where(np.array(X_BIN)==0, 1, 0)

  Y_AND = [0, 0, 0, 1]  # y
  Y_OR =  [0, 1, 1, 1]  # y
  Y_XOR = [0, 1, 1, 0]  # y

  pctBC = Perceptron(early_stopping=False, patience=5)
  ```

  Train & Test
  - AND gates
  ```
  >>> pctBC.fit(X=X_BIN, y=Y_AND) # AND gates
  >>> pctBC.predict(X=X_BIN_V)
  [1 0 0 0]
  ```
  - OR gates
  ```
  >>> pctBC.fit(X=X_BIN, y=Y_OR) # OR gates
  >>> pctBC.predict(X=X_BIN_V)
  [1 1 1 0]
  ```
  - XOR gates [1]
  ```
  >>> pctBC.fit(X=X_BIN, y=Y_XOR) # XOR gates
  >>> pctBC.predict(X=X_BIN)
  [1 1 1 1]
  ```
  - XOR gates [2]
  ```
  >>> pctBC.fit(X=X_BIN, y=Y_XOR) # XOR gates
  >>> pctBC.predict(X=X_BIN_V)
  [0 0 0 0]
  ```
  ***NOTE*** : The model cannot accurately predict XOR gates.
  
  Perceptron limits
  -----------------
  The XOR problem is a classic example of non-linearly separable problems.\n\n
  `xor` function : [`XOR(x1, x2) = AND(NOT(AND(x1, x2), OR(x1, x2)))`](https://ichi.pro/fr/perceptrons-fonctions-logiques-et-probleme-xor-61342067554570#:~:text=la%20fonction%20XOR%3F-,SOLUTION%3A,-def%20XOR_net()
  ***REASONS*** : Datasets only made of {0, 1} are not linearly separable. \n\n
    To understand why a single-layer perceptron cannot solve the XOR problem, it is also essential to understand the XOR logical function.\n\n
    The XOR function is a logical function with two variables, AND (x1, x2), which is a function with two variables with binary inputs and outputs. 
    A single-layer perceptron, which computes a weighted sum of the inputs and applies a step function, cannot solve non-linearly separable problems like XOR.
  ***CONCLUSION*** : 
    The XOR problem highlights the limitations of single-layer perceptron models and led to the development of more complex neural network architectures, such as the multilayer perceptron (MLP), which includes one or more hidden layers between the input and output layers. 
    These hidden layers allow the network to learn non-linear decision boundaries by transforming the input space into a higher-dimensional space where the classes become linearly separable.
  
  References
  ----------
  To learn more about ``Perceptron``, consult these references
  - [What is Perceptron - GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/what-is-perceptron-the-simplest-artificial-neural-network/)
  - [Deep Learning : Perceptrons simples et multicouches](https://eric.univ-lyon2.fr/ricco/cours/slides/reseaux_neurones_perceptron.pdf)
  - [Perceptrons, fonctions logiques et problème XOR](https://ichi.pro/fr/perceptrons-fonctions-logiques-et-probleme-xor-61342067554570)

  """

  def __init__(self, lr=0.001, n_iter=1000, early_stopping=True, patience=10, validation_split=0.2, shuffle=True, random_state=None):
    self._lr = lr
    self._n_iter = n_iter
    self._early_stopping = early_stopping
    self._patience = patience
    self._validation_fraction = validation_split
    self._shuffle = shuffle
    self._random_state = random_state

    self._weights : np.ndarray
    self._bias : float # b ∈ R
    self.trained_epochs = 0
    self._trained = False

  def fit(self, X:list|np.ndarray, y:list|np.ndarray):
    """
    Fit the model to the training data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    y : array-like of shape (n_samples,)
        Labels.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If input data is invalid (e.g., mismatched shapes,
        non-positive learning rate, etc.).

    """
    # Convert inputs

    X = np.array(X)
    y = np.array(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)


    # Validation checks

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must be the same.")

    if X.shape[1] == 0:
        raise ValueError("Input data must have at least one feature.")

    if self._lr <= 0:
        raise ValueError("Learning rate must be positive.")

    if self._n_iter <= 0:
        raise ValueError("Number of iterations must be positive.")

    if self._early_stopping and (
      self._patience <= 0 or
      self._validation_fraction <= 0 or
      self._validation_fraction >= 1
    ):
      raise ValueError("Invalid early stopping parameters.")


    # Random seed for reproducibility
    rng = np.random.default_rng(self._random_state)


    # Train / Validation split
    if self._early_stopping:
      X_train, y_train, X_val, y_val = split_train_validation( X, y, seed=self._random_state, validation_fraction=self._validation_fraction)
    else:
      X_train, y_train = X, y
      X_val, y_val = None, None


    # Init parameters
    self._weights = rng.standard_normal(size=X_train.shape[1]) * 0.01
    self._bias = 0.0

    if self._early_stopping:
      best_error = float('inf')
      best_weights = self._weights.copy()
      best_bias = self._bias
      no_improve = 0


    # Training loop

    for epoch in range(self._n_iter):

      if self._shuffle:
        idx = rng.permutation(X_train.shape[0])
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
      else:
        X_shuffled, y_shuffled = X_train, y_train

      # forward pass (train)
      y_pred = np.dot(X_shuffled, self._weights) + self._bias
      y_pred_class = np.where(y_pred >= 0, 1, 0)
      error = y_shuffled - y_pred_class

      # update
      self._weights += self._lr * np.dot(X_shuffled.T, error)
      self._bias += self._lr * np.sum(error)

      # Early stopping 

      if self._early_stopping and X_val is not None:

        # validation loss
        y_val_pred = np.dot(X_val, self._weights) + self._bias
        y_val_pred_class = np.where(y_val_pred >= 0, 1, 0)
        val_error = np.mean( y_val != y_val_pred_class ) # classification error

        # check improvement
        if val_error < best_error :
          best_error = val_error
          best_weights = self._weights.copy()
          best_bias = self._bias
          no_improve = 0
        else:
          no_improve += 1
        
        if no_improve >= self._patience:
          break
      self.trained_epochs = epoch + 1

    # Restore best parameters
    if self._early_stopping:
      self._weights = best_weights
      self._bias = best_bias
    self._trained = True
  
  def predict(self, X:list|np.ndarray):
    """
    Predict class labels for samples in X.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data. Must have the same number of features as the training data.
    Returns
    -------
    y_pred : np.ndarray of shape (n_samples,)
        Predicted class labels (0 or 1).
    Raises 
    ------
    ValueError
        If the model has not been trained yet, or if the input data is invalid (e.g., wrong number of features).
    """

    if not self._trained:
      raise ValueError("The model must be trained before making predictions.")

    # Convert input
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # verify input shape
    if X.shape[1] != self._weights.shape[0]:
        raise ValueError(f"Input data must have {self._weights.shape[0]} features.")
    
    y_reg = np.dot(X, self._weights) + self._bias
    y_pred = np.where(y_reg >= 0, 1, 0)
    return y_pred


  def score(self, X: np.ndarray, y: np.ndarray) -> float:
      """
      Return the accuracy of the model on the provided data
      
      Parameters:
      -----------
      X : np.ndarray of shape (n_samples, n_features)
          Test data
      y : np.ndarray of shape (n_samples,)
          True labels
          
      Returns:
      --------
      accuracy : float
          Model accuracy
      """
      if not self._trained:
          raise ValueError("The model must be trained before calculating its score.")
          
      y_pred = self.predict(X)
      return np.mean(y_pred == y)
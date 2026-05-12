import numpy as np

from .data_split import split_train_validation


class Perceptron:
  """
  **Perceptron for *`binary classification`***
  
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
  Create an `Self@Perceptron` object

  ```
  from perceptro import Perceptron
  pctBC = Perceptron(learning_rate=0.1, n_iter=1000)
  ```

  ```
  ''' Simple python demo '''
  Z : list[float] # z : float
  y : list[int|float]
  for z in Z:
    y[i] = 1 if z >= 0 else 0
  ```
  
  References
  ----------
  - https://www.geeksforgeeks.org/deep-learning/what-is-perceptron-the-simplest-artificial-neural-network/

  """

  def __init__(self, lr=0.001, n_iter=1000, early_stopping=False, patience=10, validation_split=0.2, shuffle=True, random_state=None):
    """
    Create an `Self@Perceptron` object
    """
    self.lr = lr
    self.n_iter = n_iter
    self.early_stopping = early_stopping
    self.patience = patience
    self.validation_fraction = validation_split
    self.shuffle = shuffle
    self.weights : np.ndarray
    self.bias : float # b ∈ R
    self.random_state = random_state
    self.trained_epochs = 0
    self.trained = False

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

    if self.lr <= 0:
        raise ValueError("Learning rate must be positive.")

    if self.n_iter <= 0:
        raise ValueError("Number of iterations must be positive.")

    if self.early_stopping and (
      self.patience <= 0 or
      self.validation_fraction <= 0 or
      self.validation_fraction >= 1
    ):
      raise ValueError("Invalid early stopping parameters.")


    # Random seed for reproducibility
    rng = np.random.default_rng(self.random_state)


    # Train / Validation split
    if self.early_stopping:
      X_train, y_train, X_val, y_val = split_train_validation( X, y, seed=self.random_state, validation_fraction=self.validation_fraction)
    else:
      X_train, y_train = X, y
      X_val, y_val = None, None


    # Init parameters
    self.weights = rng.standard_normal(size=X_train.shape[1]) * 0.01
    self.bias = 0.0

    if self.early_stopping:
      best_error = float('inf')
      best_weights = self.weights.copy()
      best_bias = self.bias
      no_improve = 0


    # Training loop

    for epoch in range(self.n_iter):

      if self.shuffle:
        idx = rng.permutation(X_train.shape[0])
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
      else:
        X_shuffled, y_shuffled = X_train, y_train

      # forward pass (train)
      y_pred = np.dot(X_shuffled, self.weights) + self.bias
      y_pred_class = np.where(y_pred >= 0, 1, 0)
      error = y_shuffled - y_pred_class

      # update
      self.weights += self.lr * np.dot(X_shuffled.T, error)
      self.bias += self.lr * np.sum(error)

      # Early stopping 

      if self.early_stopping and X_val is not None:

        # validation loss
        y_val_pred = np.dot(X_val, self.weights) + self.bias
        y_val_pred_class = np.where(y_val_pred >= 0, 1, 0)
        val_error = np.mean( y_val != y_val_pred_class ) # classification error

        # check improvement
        if val_error < best_error :
          best_error = val_error
          best_weights = self.weights.copy()
          best_bias = self.bias
          no_improve = 0
        else:
          no_improve += 1
        
        if no_improve >= self.patience:
          break
      self.trained_epochs = epoch + 1

    # Restore best parameters
    if self.early_stopping:
      self.weights = best_weights
      self.bias = best_bias
    self.trained = True
  
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

    if not self.trained:
      raise ValueError("The model must be trained before making predictions.")

    # Convert input
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # verify input shape
    if X.shape[1] != self.weights.shape[0]:
        raise ValueError(f"Input data must have {self.weights.shape[0]} features.")
    
    y_reg = np.dot(X, self.weights) + self.bias
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
      if not self.trained:
          raise ValueError("The model must be trained before calculating its score.")
          
      y_pred = self.predict(X)
      return np.mean(y_pred == y)
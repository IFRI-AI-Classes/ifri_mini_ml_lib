import pytest
import pandas as pd
import numpy as np

# IRIS dataset
IRIS = pd.read_csv("iris_.csv") # cd %/tests/
X_IRIS, y_iris = IRIS.drop(columns=["class"]), IRIS["class"]
X, y = X_IRIS[y_iris==2], y_iris[y_iris==2]

# X_train_test
#                 x1  x2        |
X_BIN = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
X_BIN_V = np.where(X_BIN==0, 1, 0)
# y_train_test
Y_AND = np.array([0, 0, 0, 1])  # y
Y_OR =  np.array([0, 1, 1, 1])  # y
Y_XOR = np.array([0, 1, 1, 0])  # y


from ifri_mini_ml_lib.neural_networks import Perceptron
from ifri_mini_ml_lib.preprocessing.preparation import DataSplitter
splitter = DataSplitter(seed=42)


class TestPerceptron:
  
  def test_all_process(self):
    """ Test of all process with differents hyperparameters """
    
    # tests with IRIS dataset
    X_train, X_test, y_train, y_test = splitter.train_test_split(X, y)
    INTERVAL = {  # partial INTERVAL of hyperparameters
      "n_iter": range(50, 1054, 250),
      "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1e-0],
      "patience": [5, 10],
      "shuffle": [False, True],
      "earling_stopping": [False, True],
      "random_state": [42, None],
      "validation_split": [0.1, 0.2],
    } # - combinations tests
    all_diff_perceptron_prediction = []

    # - Iterate only over the impacting hyperparameters
    for n_iter in INTERVAL["n_iter"]: # 4
      for lr in INTERVAL["learning_rate"]: # 5
        for patience in INTERVAL["patience"]: # 2
          for shuffle in INTERVAL["shuffle"]: # 2
            for earS in INTERVAL["earling_stopping"]: # 2
              perceptron = Perceptron(lr=lr, n_iter=n_iter,
                                      early_stopping=earS,
                                      patience=patience,
                                      shuffle=shuffle)
              # initialization tests
              assert perceptron.lr == lr
              assert perceptron.early_stopping == earS
              assert perceptron.patience == patience
              assert perceptron.validation_fraction in INTERVAL["validation_split"]
              assert perceptron.shuffle == shuffle
              assert perceptron.random_state in INTERVAL["random_state"]

              # train and check constraints
              perceptron.fit(X_train, y_train)
              assert perceptron.trained_epochs > 0 # iter
              assert perceptron.weights.shape[0] == X_train.shape[1] # check if dimensions elements (weights: (1,), X_train: (n,m)) matches
              assert perceptron.trained is True

              # predict and check labels shapes
              y_pred = perceptron.predict(X_test)
              assert y_pred.shape == y_test.shape
              all_diff_perceptron_prediction.append(y_pred)

    # after all (this), check conformity (or non) of predictions according to variations
    # NOTE: 
    # - all y are not the same prediction means the model is performent
    # - all y aren't the same means that the model is not performent
    assert not np.all( np.array(all_diff_perceptron_prediction) == all_diff_perceptron_prediction[0] )
    
  def test_model_limit(self):
    """ Test of `Perceptron` limits on `XOR` problem """
    perceptron = Perceptron(n_iter=10) # explicit | # test min iteration on logic gates
    X_train, X_test, y_train, y_test = X_BIN_V, X_BIN, Y_XOR, Y_XOR
    
    perceptron.fit(X_train, y_train)
    y_xor_pred = perceptron.predict(X_test)

    assert y_xor_pred is not y_test # the model fail
  
  def test_predict_without_training(self):
    """ Test of making predictions without training the model """
    perceptron = Perceptron()

    with pytest.raises(ValueError):
      perceptron.predict(X_BIN)

  def test_score(self):
    """ Test of perceptron score """
    X_train, X_test, y_train, y_test = X_BIN, X_BIN_V, Y_AND, np.where(Y_AND==0, 1, 0)
    perceptron = Perceptron()

    perceptron.fit(X_train, y_train)
    perceptron_score = perceptron.score(X_test, y_test)
    assert 0 <= perceptron_score <= 1
    
    y_pred = perceptron.predict(X_test)
    self_score = np.mean(y_pred == y_test)
    assert perceptron_score == self_score
    assert perceptron_score >= 0.5 # model performances

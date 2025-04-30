### Creating an MLP instance

```
python

model = MLP(
    hidden_layer_sizes=(100,),  # Size of hidden layers
    activation="relu",          # Activation function
    solver="sgd",               # Optimization algorithm
    alpha=0.0001,               # L2 regularization
    batch_size=32,             # Mini-batch size
    learning_rate=0.001,       # Learning rate
    max_iter=200,              # Maximum number of iterations
    shuffle=True,              # Shuffle data
    random_state=None          # Random seed
)

```
Parameters:

- *hidden_layer_sizes*: tuple of integers, default (100,)
The sizes of the hidden layers


- *activation*: str, default "relu"
Activation function: 'sigmoid', 'relu', 'tanh', 'leaky_relu'


- *solver*: str, default "sgd"
Optimization algorithm: 'sgd', 'adam', 'rmsprop', 'momentum'

  
- *alpha*: float, default 0.0001
L2 regularization coefficient


- *batch_size*: int, default 32
Size of mini-batches for training


- *learning_rate*: float, default 0.001
Learning rate for optimizers


- *max_iter*: int, default 200
Maximum number of training epochs


- *shuffle*: bool, default True
If True, shuffle data at each epoch


- *random_state*: int or None, default None
Controls random weight initialization


- *beta1*: float, default 0.9
Exponential decay parameter for Adam (first moment)


- *beta2*: float, default 0.999
Exponential decay parameter for Adam (second moment)


- *epsilon*: float, default 1e-8
Value to avoid division by zero in optimizers


- *momentum*: float, default 0.9
Parameter for momentum optimizer

### Training the Model


```
python
# X_train : training data, shape (n_samples, n_features)
# y_train : training labels, shape (n_samples,)
model.fit(X_train, y_train)
```

### Prédiction

```
python
# Predict class
y_pred = model.predict(X_test)

# Predict probabilities for each class
probas = model.predict_proba(X_test)
```

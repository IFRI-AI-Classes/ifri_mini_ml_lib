import numpy as np 
from .utils import clone

class BaggingRegressor:
    """
    Description:
        BaggingRegressor is an implementation of the Bagging (Bootstrap Aggregating) technique for regression tasks.
        It trains multiple copies of a base model on different bootstrap samples of the training set, and 
        aggregates predictions by averaging them.

    Args:
        base_model (object): A regression model implementing `fit()` and `predict()` methods.
        n_estimators (int): Number of models to train. Default is 10.
        random_state (int, optional): Seed for reproducibility. Default is None.
        pretrained_models (list, optional): A list of already trained models to use instead of training new ones.

    Returns:
        None

    Example:
        Case 1 - base model which need fitting
        >>> model = BaggingRegressor(base_model=DecisionTreeRegressor(), n_estimators=5, random_state=123)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)

        Case 2 - base models already trained which don't need fitting
        >>> trained_models = [trained_model1, trained_model2, trained_model3]
        >>> model = BaggingRegressor(pretrained_models=trained_models)
        >>> y_pred = model.predict(X_test)
    """

    def __init__(self, base_model= None, n_estimators=10, random_state=None, pretrained_models = None):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = pretrained_models if pretrained_models is not None else []
        self.pretrained = pretrained_models is not None

    def fit(self, X, y):
        """
        Description:
            Trains multiple instances of the base model on bootstrap samples of the training data.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target vector of shape (n_samples,).

        Returns:
            None

        Example:
            >>> model.fit(X_train, y_train)
        """
        if self.pretrained:
            for model in self.models:
                if not hasattr(model, 'predict'):
                    raise ValueError("Each pretrained model must implement the predict() method.")
            return #We skip training
        

        X, y = np.array(X), np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
            
        if not (hasattr(self.base_model, 'fit') and hasattr(self.base_model, 'predict')):
            raise ValueError("The base model must implement both fit() and predict() methods.")
        
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Échantillonner avec remise
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            #model = self.base_model
            model = clone(self.base_model)            
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        """
        Description:
            Predicts target values for given input samples by averaging the predictions of all models.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Averaged predictions for all samples.

        Example:
            >>> y_pred = model.predict(X_test)
        """
        X = np.array(X)
        predictions = np.zeros((self.n_estimators, X.shape[0]))
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X)

        return np.mean(predictions, axis=0)

class BaggingClassifier:
    """
    Description:
        BaggingClassifier is an implementation of the Bagging (Bootstrap Aggregating) technique for classification tasks.
        It trains multiple copies of a base classifier on bootstrap samples of the training set, and 
        aggregates predictions using majority voting.

    Args:
        base_model (object): A classification model implementing `fit()` and `predict()` methods.
        n_estimators (int): Number of models to train. Default is 10.
        random_state (int, optional): Seed for reproducibility. Default is None.
        pretrained_models (list, optional): A list of already trained models to use instead of training new ones.

    Returns:
        None

    Example:
        Case 1 - base model which need fitting
        >>> model = BaggingClassifier(base_model=DecisionTreeClassifier(), n_estimators=5, random_state=123)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)

        Case 2 - base models already trained which don't need fitting
        >>> trained_models = [trained_model1, trained_model2, trained_model3]
        >>> model = BaggingClassifier(pretrained_models=trained_models)
        >>> y_pred = model.predict(X_test)

    """

    def __init__(self, base_model=None, n_estimators=10, random_state = None, pretrained_models = None):
        
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = pretrained_models if pretrained_models is not None else []
        self.pretrained = pretrained_models is not None


    def fit(self, X, y):
        """
        Description:
            Trains multiple instances of the base model on bootstrap samples of the training data.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).
            y (array-like): Target vector of shape (n_samples,).

        Returns:
            None

        Example:
            >>> model.fit(X_train, y_train)
        """

        if self.pretrained:
            for model in self.models:
                if not hasattr(model, 'predict'):
                    raise ValueError("Each pretrained model must implement the predict() method.")
            return  # Skip training
        
        X, y = np.array(X), np.array(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
            
        if not (hasattr(self.base_model, 'fit') and hasattr(self.base_model, 'predict')):
            raise ValueError("The base model must implement both fit() and predict() methods")
        
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Échantillonner avec remise
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            model = clone(self.base_model)
            
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        """
        Description:
            Predicts target classes for given input samples using majority voting across all models.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels for all samples.

        Example:
            >>> y_pred = model.predict(X_test)
        """

        predictions = np.zeros((self.n_estimators, X.shape[0]))
        for i, model in enumerate(self.models):
            predictions[i] = model.predict(X)

        #return np.array([np.argmax(np.bincount(predictions[:, i].astype(int))) for i in range(X.shape[0])])
        from scipy.stats import mode
        return mode(predictions, axis=0)[0].ravel()
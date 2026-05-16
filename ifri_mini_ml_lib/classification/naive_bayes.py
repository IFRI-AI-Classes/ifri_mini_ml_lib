import numpy as np

class NaiveBayes:
    """
    Gaussian Naive Bayes (GNB) algorithm for binary and multi-class classification.
    
    Description:
        Implements the Gaussian Naive Bayes algorithm from scratch. This classifier
        assumes that the continuous features follow a normal (Gaussian) distribution
        and that features are independent given the class (the "naive" assumption).
        It includes internal feature standardization (Z-score) for numerical stability.
    
    Examples:
        >>> nb = NaiveBayes()
        >>> nb.fit(X_train, y_train)
        >>> predictions = nb.predict(X_test)
        >>> accuracy = np.mean(predictions == y_test)
    """

    def __init__(self):
        """
        Initializes the NaiveBayes classifier and its internal storage.
        """
        self.mean = {}     # Stores mean of each feature per class
        self.var = {}      # Stores variance of each feature per class
        self.priors = {}   # Stores prior probability P(C) of each class
        self.classes = []  # List of unique class labels
        self.scaler_mean = None # Global mean for standardization
        self.scaler_std = None  # Global standard deviation for standardization

    def fit(self, X, y):
        """
        Trains the model by calculating statistics and prior probabilities.
        
        Description:
            Computes the global mean and standard deviation for feature scaling,
            then calculates the mean, variance, and prior probability for each 
            unique class in the standardized training dataset.
        
        Args:
            X (array-like): Training samples of shape (n_samples, n_features).
            y (array-like): Target labels of shape (n_samples,).
        """
        # Feature Scaling (Standardization)
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0) + 1e-9
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        self.classes = np.unique(y)
        
        # Calculate parameters for each class
        for c in self.classes:
            X_c = X_scaled[y == c]
            self.priors[c] = X_c.shape[0] / float(len(y))
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9

    def predict(self, X):
        """
        Predicts the class labels for the provided test data.
        
        Description:
            Standardizes the input data using parameters learned during fit,
            calculates the log-likelihood for each class using the Gaussian PDF,
            and returns the class with the highest posterior probability.
        
        Args:
            X (array-like): Test samples of shape (n_samples, n_features).
            
        Returns:
            list: Predicted class labels for each input sample.
        """
        # Apply the same scaling as the training data
        X = (X - self.scaler_mean) / self.scaler_std
        
        predictions = []
        for x in X:
            log_probs = {}
            
            for c in self.classes:
                # Prior probability log(P(C))
                log_prior = np.log(self.priors[c])
                
                # Log-likelihood using the Gaussian PDF formula
                # log(P(X|C)) = sum(-0.5 * log(2 * pi * var) - 0.5 * (x - mean)^2 / var)
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.var[c])) \
                                 - 0.5 * np.sum(((x - self.mean[c]) ** 2) / self.var[c])
                
                log_probs[c] = log_prior + log_likelihood
            
            # Select the class with the maximum log-posterior probability
            best_class = max(log_probs, key=log_probs.get)
            predictions.append(best_class)
            
        return predictions
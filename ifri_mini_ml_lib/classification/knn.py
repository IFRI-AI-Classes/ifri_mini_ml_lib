import numpy as np

class KNN:
    """
    K-Nearest Neighbors (KNN) algorithm for classification and regression.
    
    Description:
        Implements the KNN algorithm which can be used for both classification
        and regression tasks. The algorithm finds the k closest training examples
        to make predictions for new data points.
    
    Args:
        k (int, optional): Number of nearest neighbors to consider. Default is 3.
        task (str, optional): Type of task - either 'classification' or 'regression'.
                            Default is 'classification'.
    
    Examples:
        >>> # For classification
        >>> knn = KNN(k=5, task='classification')
        >>> knn.fit(X_train, y_train)
        >>> predictions = knn.predict(X_test)
        
        >>> # For regression
        >>> knn_reg = KNN(k=3, task='regression')
        >>> knn_reg.fit(X_train, y_train)
        >>> predictions = knn_reg.predict(X_test)
    """
    def __init__(self, k=3, task='classification'):
        """
        Initializes the KNN classifier/regressor.
        
        Args:
            k (int): Number of neighbors to use.
            task (str): Type of task ('classification' or 'regression').
        """
        self.k = k
        self.task = task 
    
    def fit(self, X, y):
        """
        Stores the training dataset for later predictions.
        
        Description:
            This method doesn't actually "train" the model in the traditional sense
            since KNN is a lazy learner. It simply stores the training data
            to use during the prediction phase.
        
        Args:
            X (array-like): Training samples of shape (n_samples, n_features).
            y (array-like): Target values of shape (n_samples,).
        """
        self.x_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X):
        """
        Predicts the target values for the provided data.
        
        Description:
            For each input sample, finds the k-nearest neighbors in the training set
            and returns either the most common class (classification) or the
            average value (regression).
        
        Args:
            X (array-like): Test samples of shape (n_samples, n_features).
            
        Returns:
            list: Predicted target values for each input sample.
        """
        return [self._predict(x) for x in X]
    
    def _predict(self, x):
        """
        Helper method to predict the target for a single sample.
        
        Description:
            Computes distances to all training samples, finds the k-nearest neighbors,
            and makes a prediction based on their labels/values.
        
        Args:
            x (array-like): A single input sample of shape (n_features,).
            
        Returns:
            int/float: Predicted class label (classification) or value (regression).
        """
        # Compute Euclidean distances between x and all training samples
        distances = np.linalg.norm(self.x_train - x, axis=1)
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels/values of the nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        if self.task == 'regression':
            # Return mean value for regression tasks
            return np.mean(k_nearest_labels)
        else:
            # Return most frequent class for classification tasks
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            
            # In case of ties, return the class with the smallest index (stable behavior)
            max_count_indices = np.where(counts == np.max(counts))[0]
            return unique_labels[max_count_indices[0]]

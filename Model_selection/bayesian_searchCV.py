import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from cross_validation import k_fold_cross_validation

class GaussianProcess:
    """
    Gaussian process with RBF (Radial Basis Function) kernel.
    Used to model the objective function in Bayesian optimization.
    """

    def __init__(self, kernel_var=1.0, length_scale=1.0, noise=1e-6):
        """
        Description:
            Initialize the Gaussian process model with kernel parameters.
            
        Args:
            kernel_var: Variance of the RBF kernel (default: 1.0)
            length_scale: Length scale parameter of the kernel (default: 1.0)
            noise: Regularization noise added to diagonal (default: 1e-6)
            
        Returns:
            None
            
        Example:
            >>> gp = GaussianProcess(kernel_var=2.0, length_scale=0.5)
        """
        self.kernel_var = kernel_var
        self.length_scale = length_scale
        self.noise = noise

    def rbf_kernel(self, X1, X2):
        """
        Description:
            Compute the covariance matrix between two sets of points using RBF kernel.
            
        Args:
            X1: First set of points, array of shape (n_samples_1, n_features)
            X2: Second set of points, array of shape (n_samples_2, n_features)
            
        Returns:
            Covariance matrix of shape (n_samples_1, n_samples_2)
            
        Example:
            >>> X1 = np.array([[1, 2], [3, 4]])
            >>> X2 = np.array([[5, 6], [7, 8]])
            >>> K = gp.rbf_kernel(X1, X2)
        """
        dists = cdist(X1, X2, 'sqeuclidean')  # squared euclidean distance
        return self.kernel_var * np.exp(-0.5 * dists / self.length_scale**2)

    def fit(self, X_train, y_train):
        """
        Description:
            Train the Gaussian Process model with given data.
            
        Args:
            X_train: Training feature data, array of shape (n_samples, n_features)
            y_train: Training target values, array of shape (n_samples,)
            
        Returns:
            None
            
        Example:
            >>> gp.fit(X_train, y_train)
        """
        self.X_train = X_train
        self.y_train = y_train
        K = self.rbf_kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test):
        """
        Description:
            Predict mean and standard deviation for new test points.
            
        Args:
            X_test: Test points, array of shape (n_samples, n_features)
            
        Returns:
            tuple: (mean, std_dev) where both are arrays of shape (n_samples,)
            
        Example:
            >>> mu, sigma = gp.predict(X_test)
        """
        K_s = self.rbf_kernel(self.X_train, X_test)
        K_ss = self.rbf_kernel(X_test, X_test) + self.noise * np.eye(len(X_test))
        mu = K_s.T @ self.K_inv @ self.y_train
        cov = K_ss - K_s.T @ self.K_inv @ K_s
        return mu, np.sqrt(np.diag(cov))


def expected_improvement(X, gp, y_min, xi=0.01):
    """
    Description:
        Calculate Expected Improvement (EI) acquisition function for given candidate points.
        
    Args:
        X: Candidate points to evaluate, array of shape (n_samples, n_features)
        gp: Trained GaussianProcess model
        y_min: Best objective value observed so far
        xi: Exploration parameter, higher values encourage exploration (default: 0.01)
        
    Returns:
        array: Expected improvement values for each point, shape (n_samples,)
        
    Example:
        >>> X_candidates = np.random.uniform(0, 1, (100, 2))
        >>> ei = expected_improvement(X_candidates, gp, best_score, xi=0.01)
    """
    mu, sigma = gp.predict(X)
    with np.errstate(divide='warn'):
        imp = y_min - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0  # avoid division by zero
    return ei



class BayesianSearchCV:
    """
    Bayesian optimization with cross-validation for hyperparameter search.
    Uses a Gaussian process to model the objective function and Expected Improvement for acquisition.
    """

    def __init__(self, estimator, param_bounds, scoring, stratified=None, n_iter=20, init_points=5, cv=5):
        """
        Description:
            Initialize the Bayesian search optimization.
            
        Args:
            estimator: ML model to optimize (must implement fit and predict methods)
            param_bounds: Dictionary of hyperparameter bounds, where keys are parameter 
                         names and values are tuples (min_bound, max_bound)
            scoring: Evaluation metric function
            stratified: Whether to use stratified CV (default: None)
            n_iter: Number of optimization iterations after initialization (default: 20)
            init_points: Number of random points for initialization (default: 5)
            cv: Number of cross-validation folds (default: 5)
            
        Returns:
            None
            
        Example:
            >>> from ifri_mini_lib.supervised.classification import SVC
            >>> from ifri_mini_lib.metrics import accuracy_score
            >>> model = SVC()
            >>> param_bounds = {'C': (0.1, 100), 'gamma': (0.001, 1.0)}
            >>> bo = BayesianSearchCV(model, param_bounds, accuracy_score, n_iter=30)
        """
        self.estimator = estimator
        self.param_bounds = param_bounds
        self.n_iter = n_iter
        self.init_points = init_points
        self.cv = cv
        self.scoring = scoring
        self.stratified = stratified
        self.X_obs = []  # list of tested hyperparameter vectors
        self.y_obs = []  # corresponding scores
        self.gp = GaussianProcess()

    def _sample_params(self):
        """
        Description:
            Generate random hyperparameter vector within specified bounds.
            
        Args:
            None
            
        Returns:
            array: Random hyperparameter vector
            
        Example:
            >>> random_params = bo._sample_params()
        """
        return np.array([np.random.uniform(low, high) for (low, high) in self.param_bounds.values()])

    def _dict_from_vector(self, x_vector):
        """
        Description:
            Convert hyperparameter vector to dictionary with parameter names.
            
        Args:
            x_vector: Vector of hyperparameter values
            
        Returns:
            dict: Dictionary mapping parameter names to values
            
        Example:
            >>> params_dict = bo._dict_from_vector([1.0, 0.1])
        """
        return dict(zip(self.param_bounds.keys(), x_vector))


    def _evaluate(self, X, y, x):
        """
        Description:
            Evaluate a set of hyperparameters using cross-validation.
            
        Args:
            X: Input data, array of shape (n_samples, n_features)
            y: Target values, array of shape (n_samples,)
            x: Hyperparameters to evaluate, either vector or dictionary
            
        Returns:
            float: Cross-validation score
            
        Example:
            >>> score = bo._evaluate(X_train, y_train, {'C': 10, 'gamma': 0.1})
        """
        if not isinstance(x, dict):
            x = self._dict_from_vector(x)

        # Special handling for integer parameters
        if "n_neighbors" in x:
            x["n_neighbors"] = int(round(x["n_neighbors"]))
            if x["n_neighbors"] < 1:
                raise ValueError("n_neighbors must be a positive integer")

        # Apply parameters to model
        self.estimator.set_params(**x)
        mean_score, _ = k_fold_cross_validation(self.estimator, X, y, self.scoring, self.stratified, k=self.cv)
        return mean_score

    def _suggest(self, n_candidates=100):
        """
        Description:
            Suggest next hyperparameters to evaluate using Expected Improvement.
            
        Args:
            n_candidates: Number of random candidates to generate (default: 100)
            
        Returns:
            array: Vector of suggested hyperparameter values
            
        Example:
            >>> next_params = bo._suggest()
        """
        X_candidates = np.array([self._sample_params() for _ in range(n_candidates)])
        ei = expected_improvement(X_candidates, self.gp, y_min=np.max(self.y_obs))
        return X_candidates[np.argmax(ei)]

    def fit(self, X, y):
        """
        Description:
            Run Bayesian optimization to find optimal hyperparameters.
            
        Args:
            X: Input data, array of shape (n_samples, n_features)
            y: Target values, array of shape (n_samples,)
            
        Returns:
            self: The instance itself
            
        Example:
            >>> bo.fit(X_train, y_train)
            >>> best_params = bo.best_params_
            >>> best_score = bo.best_score_
        """
        X, y = np.array(X), np.array(y)

        # Initial phase: random points
        for _ in range(self.init_points):
            x = self._sample_params()
            y_score = self._evaluate(X, y, x)
            self.X_obs.append(x)
            self.y_obs.append(y_score)

        # Optimization loop
        for i in range(self.n_iter):
            self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))
            x_next = self._suggest()
            y_next = self._evaluate(X, y, x_next)
            self.X_obs.append(x_next)
            self.y_obs.append(y_next)
            print(f"[{i+1}/{self.n_iter}] Score = {y_next:.4f}")

        # Get best parameters
        best_idx = np.argmin(self.y_obs)  # minimization if score is an error
        self.best_params_ = self._dict_from_vector(self.X_obs[best_idx])
        self.best_score_ = self.y_obs[best_idx]

    def get_best_params(self):
        """
        Description:
            Return the best hyperparameters found during optimization.
            
        Args:
            None
            
        Returns:
            dict: Dictionary with best hyperparameter values
            
        Example:
            >>> best_params = bo.get_best_params()
            >>> print(best_params)
            {'C': 10.0, 'gamma': 0.01}
        """
        return self.best_params_
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .pca import PCA


class TSNE:

    """
    Implementation of the t-Distributed Stochastic Neighbor Embedding (t-SNE) algorithm
    for dimensionality reduction and data visualization.

    Attributes:
        n_components (int): Dimension of the projection space (2 or 3)
        perplexity (float): Effective number of local neighbors (typically between 5 and 50)
        early_exaggeration (float): Initial exaggeration factor for optimization
        learning_rate (float): Learning rate for gradient descent
        n_iter (int): Maximum number of iterations
        embedding_ (np.ndarray): Projection result (shape: [n_samples, n_components])
        kl_divergence_ (float): Last KL divergence value
        n_iter_ (int): Actual number of iterations performed
    """

    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0, 
                 learning_rate=200.0, n_iter=1000, init = 'pca', min_grad_norm=1e-7, random_state=None, verbose=0):

        """
        Description:
            Initializes the t-SNE parameters.

        Args:
            n_components (int): Output dimension (default: 2)
            perplexity (float): Perplexity (default: 30.0)
            early_exaggeration (float): Initial exaggeration factor (default: 12.0)
            learning_rate (float): Learning rate (default: 200.0)
            init (str): Initialization method ('pca' or 'random') (default: 'pca')
            n_iter (int): Maximum number of iterations (default: 1000)
            min_grad_norm (float): Gradient norm stopping threshold (default: 1e-7)
            random_state (int): Reproducibility seed (default: None)
            verbose (int): Verbosity level (0 or 1) (default: 0)

        Example:
            >>> tsne = TSNE(n_components=2, perplexity=20)
        """

        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.init = init
        self.min_grad_norm = min_grad_norm
        self.random_state = random_state
        self.verbose = verbose
        
        # Results
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = None
        
        self.rng = np.random.RandomState(random_state)  # Random state for reproducibility
    

    def _euclidean_distance(self, X: np.ndarray) -> np.ndarray:
       
        """
        Description:
            Computes the squared Euclidean distance matrix.

        Args:
            X (np.ndarray): Input data (shape: [n_samples, n_features])

        Returns:
            np.ndarray: Squared distance matrix (shape: [n_samples, n_samples])
        """

        sum_X = np.sum(np.square(X), axis=1)
        distances = np.add(-2 * np.dot(X, X.T), sum_X).T + sum_X
        np.fill_diagonal(distances, 0.0)
        return distances
    

    def _binary_search_perplexity(self, distances, perplexity, tol=1e-5, max_iter=50) -> np.ndarray:

        """
        Description:
            Dichotomous search to find the optimal sigmas.

        Args:
            distances (np.ndarray): Distance matrix
            perplexity (float): Target perplexity
            tol (float): Convergence tolerance
            max_iter (int): Maximum iterations

        Returns:
            np.ndarray: Conditional probability matrix P
        """

        n_samples = distances.shape[0]
        P = np.zeros((n_samples, n_samples))
        beta = np.ones((n_samples, 1))
        log_perplexity = np.log(perplexity)
        
        # We ignore the diagonal (distance from oneself = 0)
        for i in range(n_samples):
            beta_min = -np.inf
            beta_max = np.inf
            dist_i = distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))]
            
            for _ in range(max_iter):
                # Calculation of conditional probabilities
                P_i = np.exp(-dist_i * beta[i])
                sum_Pi = np.sum(P_i)
                
                if sum_Pi == 0:
                    sum_Pi = 1e-8
                
                # Calculation of entropy
                H = np.log(sum_Pi) + beta[i] * np.sum(dist_i * P_i) / sum_Pi
                P_i = P_i / sum_Pi
                
                # Beta adjustment (binary precision)
                H_diff = H - log_perplexity
                if np.abs(H_diff) < tol:
                    break
                
                if H_diff > 0:
                    beta_min = beta[i].copy()
                    if beta_max == np.inf:
                        beta[i] *= 2.0
                    else:
                        beta[i] = (beta[i] + beta_max) / 2.0
                else:
                    beta_max = beta[i].copy()
                    if beta_min == -np.inf:
                        beta[i] /= 2.0
                    else:
                        beta[i] = (beta[i] + beta_min) / 2.0
            
            # Fill in the P matrix
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))] = P_i
        
        return P
    

    def _compute_joint_probabilities(self, X: np.ndarray, perplexity: float) -> np.ndarray:

        """
        Description:
            Computes the joint probabilities P in high dimensions.

        Args:
            X (np.ndarray): Original data
            perplexity (float): Desired perplexity

        Returns:
            np.ndarray: Symmetric joint probability matrix
        """

        # Calculation of square Euclidean distances
        distances = self._euclidean_distance(X)
        
        # Calculation of conditional probabilities
        P = self._binary_search_perplexity(distances, perplexity)
        
        # Symmetrization and normalization
        P = (P + P.T) / (2.0 * P.shape[0])
        P /= np.sum(P)
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_low_dimensional_probabilities(self, Y: np.ndarray) -> np.ndarray:

        """
        Description:
            Computes the low-dimensional joint probabilities q_ij according to a Student t distribution.

        Args:
            Y (np.ndarray): Current low-dimensional embedding (shape: [n_samples, n_components])

        Returns:
            np.ndarray: Probability matrix Q (shape: [n_samples, n_samples])

        Example:
            >>> q_matrix = self._compute_low_dimensional_probabilities(current_embedding)
        """

        distances = self._euclidean_distance(Y)
        inv_distances = 1.0 / (1.0 + distances)
        np.fill_diagonal(inv_distances, 0.0)
        Q = inv_distances / np.sum(inv_distances)
        Q = np.maximum(Q, 1e-12)
        return Q
    

    def _compute_gradient(self, P: np.ndarray, Q: np.ndarray, Y: np.ndarray) -> np.ndarray:

        """
        Description:
            Computes the gradient of the KL divergence with respect to the embedding coordinates.

        Args:
            P (np.ndarray): High-dimensional probability matrix
            Q (np.ndarray): Low-dimensional probability matrix
            Y (np.ndarray): Current embedding

        Returns:
            np.ndarray: Gradient matrix (same shape as Y)

        Example:
            >>> grad = self._compute_gradient(p_matrix, q_matrix, current_embedding)
        """

        diff = Y[:, None, :] - Y[None, :, :]  # Shape: (n_samples, n_samples, n_components)
        pq_diff = (P - Q)[..., None] * (1.0 / (1.0 + np.clip(self._euclidean_distance(Y), 1e-8, None)))[..., None] 
        
        return 4 * (pq_diff * diff).sum(axis=1)  # Shape: (n_samples, n_components)
    

    def _compute_kl_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:

        """
        Description:
            Computes the Kullback-Leibler divergence between the P and Q distributions.

        Args:
            P (np.ndarray): Reference probability distribution
            Q (np.ndarray): Approximate distribution

        Returns:
            float: KL divergence value (in bits)

        Example:
            >>> kl = self._compute_kl_divergence(p_matrix, q_matrix)
        """
        return np.sum(P * np.log(P / Q))
    

    def fit(self, X: np.ndarray) -> 'TSNE':

        """
        Description:
            Training the t-SNE model.

        Args:
            X (np.ndarray): Data to project (shape: [n_samples, n_features])

        Returns:
            TSNE: Trained instance

        Raises:
            ValueError: If n_samples < 3 * perplexity

        Example:
            >>> data = np.random.rand(100, 10)
            >>> tsne.fit(data)
        """

        n_samples = X.shape[0]
        
        # Data verification
        if n_samples < 3 * self.perplexity:
            raise ValueError(f"The number of samples ({n_samples}) must be at least 3 * perplexity ({3*self.perplexity})")
        
        if self.verbose:
            print("Calculation of joint probabilities P...")
        
        # Calculation of P in high dimension
        P = self._compute_joint_probabilities(X, self.perplexity)
        P_early = P * self.early_exaggeration
        
        if self.init == 'random':
            # Random initialization of Y
            Y = 1e-4 * np.random.randn(n_samples, self.n_components).astype(np.float32)
        else:
            # Initialization of Y by pca
            Y = PCA(n_component=self.n_components).fit_transform(X)
            Y = Y / np.std(Y, axis=0) * 1e-4  
        
        exaggeration_end_iter = 250

        # Variables for optimization
        velocity = np.zeros_like(Y)      # Initialization of momentum
        gains = np.ones_like(Y)
        
        if self.verbose:
            print("Optimization of embedding...")
        
        # Optimisation
        for i in range(self.n_iter):
            # Calculation of Q in low dimension
            Q = self._compute_low_dimensional_probabilities(Y)
            
            # Gradient calculation
            gradient = self._compute_gradient(P_early if i < exaggeration_end_iter else P , Q, Y)
            grad_norm = np.linalg.norm(gradient)
            
            # Update with momentum
            momentum = 0.5 if i < exaggeration_end_iter else 0.8

            # Update of gains
            gains = (gains + 0.2) * ((gradient > 0) != (velocity > 0)) + \
                    (gains * 0.8) * ((gradient > 0) == (velocity > 0))
            gains = np.clip(gains, 0.01, np.inf)

            # velocity's update with momentum
            velocity = momentum * velocity - self.learning_rate * (gains * gradient)
            
            # Update of embedding
            Y += velocity

            # Data Centering
            Y -= Y.mean(axis=0)

            # Dynamic learning rate update
            if i == exaggeration_end_iter:
                self.learning_rate *= 0.8
            
            # Reinitialization of P after early exaggeration
            if i == exaggeration_end_iter:
                P = self._compute_joint_probabilities(X, self.perplexity)
            
            # Calculation of KL divergence
            kl_div = self._compute_kl_divergence(P_early if i < exaggeration_end_iter else P, Q)
            
            # Displaying information
            if self.verbose >= 1 and i % 100 == 0:
                print(f"Iteration {i}: KL divergence = {kl_div:.4f}, Gradient norm = {grad_norm:.4f}, LR={self.learning_rate:.1f}")
                
            if grad_norm < self.min_grad_norm:
                if self.verbose:
                    print(f"Premature stop at iteration {i}: gradient norm too low")
                break
        
        # Saving results
        self.embedding_ = Y.copy()
        #self.embedding_ = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)  # Normalization
        self.kl_divergence_ = kl_div
        self.n_iter_ = i + 1
        
        return self
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        """
        Description:
            One-step data training and projection.

        Args:
            X (np.ndarray): Data to transform

        Returns:
            np.ndarray: Projected data (shape: [n_samples, n_components])

        Example:
            >>> embedding = tsne.fit_transform(data)
        """

        self.fit(X)
        assert self.embedding_.shape[1] == self.n_components, "Embedding shape mismatch"
        return self.embedding_


    @staticmethod
    def generate_test_data(n_samples=300, case='blobs', random_state=None):

        """
        Description:
            Generates test data for visualization.

        Args:
            n_samples (int): Number of samples (default: 300)
            case (str): Data type ('blobs', 'swiss_roll', or other) (default: 'blobs')
            random_state (int): Random seed (default: None)

        Returns:
            tuple: (X, y) data and labels

        Example:
            >>> X, y = TSNE.generate_test_data(case='swiss_roll')
        """

        if random_state:
            np.random.seed(random_state)
        
        if case == 'blobs':
            # Clustered data
            centers = np.array([[1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]])
            X = np.vstack([center + np.random.randn(n_samples//4, 3)*0.3 for center in centers])
            y = np.repeat(np.arange(4), n_samples//4)
        elif case == 'swiss_roll':
            # Swiss roll data
            t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
            X = np.vstack([t * np.cos(t), 10 * np.random.rand(n_samples), t * np.sin(t)]).T
            y = (t // np.pi).astype(int)
        else:
            # Linearly separable data
            X = np.random.randn(n_samples, 3)
            X[:n_samples//2] += 1
            X[n_samples//2:] -= 1
            y = np.zeros(n_samples)
            y[n_samples//2:] = 1
        
        # Normalisation
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return X, y


    @staticmethod
    def plot_results(X, y, title, ax=None):

        """
        Description:
            Views the projection results.

        Args:
            X (np.ndarray): Projected data (2D or 3D)
            y (np.ndarray): Labels for coloring
            title (str): Plot title
            ax (matplotlib.axes.Axes): Optional axis for the plot

        Example:
            >>> TSNE.plot_results(embedding, y, 't-SNE Projection')
        """

        if ax is None:
            fig = plt.figure()
            if X.shape[1] == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
        
        if X.shape[1] == 2:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.7)
        ax.set_title(title)
        ax.grid(True)


        

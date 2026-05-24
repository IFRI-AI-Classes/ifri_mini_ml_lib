import numpy as np


class Loss:
    """
    Base class for loss functions.
    
    All specific loss functions should inherit from this class and implement
    the __call__ method. Provides a common interface for L2 regularization.
    
    Methods
    -------
    __call__(y_true, y_pred)
        Compute the loss between true and predicted values.
    l2_regularization(weights, n_samples)
        Compute L2 regularization term.
    gradient(y_true, y_pred, output_activation_name)
        Compute the gradient of the loss with respect to the output logits.
    
    """
    
    def __call__(self, y_true, y_pred):
        """
        Compute the loss between true and predicted values.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values.
        y_pred : np.ndarray
            Predicted values from the model.
            
        Returns
        -------
        float
            The computed loss value.
            
        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the __call__ method.")

    def l2_regularization(self, weights, n_samples):
        """
        Compute L2 regularization term (weight decay).
        
        The L2 regularization penalizes large weights to prevent overfitting.
        Formula: (l2_alpha / (2 * n_samples)) * sum(w^2)
        
        Parameters
        ----------
        weights : list of np.ndarray or None
            List of weight matrices from the model layers.
            If None, returns 0.0 (no regularization).
        n_samples : int
            Number of samples in the batch (used for scaling).
            
        Returns
        -------
        float
            The L2 regularization term. Returns 0.0 if weights is None.
            
        Examples
        --------
        >>> loss_fn = MeanSquaredError(l2_alpha=0.01)
        >>> weights = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        >>> reg = loss_fn.l2_regularization(weights, n_samples=32)
        """
        if weights is None:
            return 0.0
        l2 = sum((w**2).sum() for w in weights)
        return l2 * self.l2_alpha / (2 * n_samples)

    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of the loss with respect to the output logits
        (the inputs to the output activation function).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.

        y_pred : np.ndarray
            Model predictions after applying the output activation function.

        output_activation_name : str or None
            Name of the output activation function
            (e.g. 'sigmoid', 'softmax', 'linear', or None).

        Returns
        -------
        np.ndarray
            Gradient δ = ∂L/∂z, where z represents the logits
            (values before the output activation function).

        Notes
        -----
        For some common activation-loss combinations such as:
        - sigmoid + binary cross-entropy
        - softmax + categorical cross-entropy

        the gradient simplifies directly to:

            y_pred - y_true

        which already corresponds to the gradient with respect to the logits.
        """
        raise NotImplementedError(
            f"Loss {self.__class__.__name__} must implement gradient()"
        )

class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function.
    
    Computes the average of squared differences between true and predicted values.
    Penalizes large errors more heavily than small ones.
    
    Formula
    -------
    MSE = (1/N) * sum((y_true - y_pred)^2)
    
    With L2 regularization:
    Loss = MSE + (l2_alpha / (2N)) * sum(w^2)
    
    Parameters
    ----------
    l2_alpha : float, default=0.0
        L2 regularization strength.
        
    Methods
    -------
    __call__(y_true, y_pred, weights=None)
        Compute MSE with optional L2 regularization.
        
    Examples
    --------
    >>> mse = MeanSquaredError(l2_alpha=0.01)
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.5, 2.0, 2.5])
    >>> loss = mse(y_true, y_pred)
    
    Notes
    -----
    - Sensitive to outliers (large errors are squared and amplified)
    - Gradient decreases as error decreases (converges smoothly)
    - Most common loss for regression tasks
    """
    
    def __init__(self, l2_alpha=0.0):
        self.l2_alpha = l2_alpha

    def __call__(self, y_true, y_pred, weights=None):
        """
        Compute Mean Squared Error with optional L2 regularization.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_pred : np.ndarray
            Predicted values.
        weights : list of np.ndarray, optional
            Model weights for L2 regularization.
            
        Returns
        -------
        float
            The total loss (MSE + L2 regularization if applicable).
        """
        mse = np.mean(np.square(y_true - y_pred))
        l2 = self.l2_regularization(weights, y_true.shape[0])
        return mse + l2

    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of MSE with respect to the output logits.

        This implementation assumes a linear output activation.
        """
        if output_activation_name != "linear":
            raise NotImplementedError(
                "Gradient for MSE is only implemented for linear output activation."
            )

        return 2.0 * (y_pred - y_true)

class MeanAbsoluteError(Loss):
    """
    Mean Absolute Error (MAE) loss function.
    
    Computes the average of absolute differences between true and predicted values.
    Robust to outliers compared to MSE.
    
    Formula
    -------
    MAE = (1/N) * sum(|y_true - y_pred|)
    
    With L2 regularization:
    Loss = MAE + (l2_alpha / (2N)) * sum(w^2)
    
    Parameters
    ----------
    l2_alpha : float, default=0.0
        L2 regularization strength. Set to 0.0 to disable regularization.
        Higher values increase penalty on large weights.
        
    Methods
    -------
    __call__(y_true, y_pred, weights=None)
        Compute MAE loss with optional L2 regularization.
        
    Examples
    --------
    >>> mae = MeanAbsoluteError(l2_alpha=0.01)
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.1, 1.9, 3.2])
    >>> loss = mae(y_true, y_pred)
    >>> print(f"MAE: {loss:.4f}")
    MAE: 0.1333
    
    Notes
    -----
    - Robust to outliers (less sensitive than MSE to large errors)
    - Gradient is constant (does not decrease for small errors)
    - Suitable for regression tasks
    """
    
    def __init__(self, l2_alpha=0.0):
        self.l2_alpha = l2_alpha

    def __call__(self, y_true, y_pred, weights=None):
        """
        Compute Mean Absolute Error with optional L2 regularization.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values, shape (n_samples,) or (n_samples, n_outputs).
        y_pred : np.ndarray
            Predicted values, same shape as y_true.
        weights : list of np.ndarray, optional
            Model weights for L2 regularization. If None, no regularization is applied.
            
        Returns
        -------
        float
            The total loss (MAE + L2 regularization if applicable).
        """
        mae = np.mean(np.abs(y_true - y_pred))
        l2 = self.l2_regularization(weights, y_true.shape[0])
        return mae + l2

    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of MAE with respect to the output logits.

        This implementation assumes a linear output activation.
        """
        if output_activation_name != "linear":
            raise NotImplementedError(
                "Gradient for MAE is only implemented for linear output activation."
            )

        
        return np.sign(y_pred - y_true)

class HuberLoss(Loss):
    """
    Huber loss function.
    
    Combines MSE and MAE: uses MSE for small errors and MAE for large errors.
    Less sensitive to outliers than MSE while maintaining smoothness near zero.
    
    Formula
    -------
    For each error e = y_true - y_pred:
    - If |e| <= delta: 0.5 * e^2  (MSE-like)
    - If |e| > delta: delta * (|e| - 0.5 * delta)  (MAE-like)
    
    With L2 regularization:
    Loss = Huber + (l2_alpha / (2N)) * sum(w^2)
    
    Parameters
    ----------
    delta : float, default=1.0
        Threshold determining the transition between quadratic and linear behavior.
        - Small delta: more robust to outliers (MAE-like)
        - Large delta: more sensitive to all errors (MSE-like)
    l2_alpha : float, default=0.0
        L2 regularization strength.
        
    Methods
    -------
    __call__(y_true, y_pred, weights=None)
        Compute Huber loss with optional L2 regularization.
        
    Examples
    --------
    >>> huber = HuberLoss(delta=1.0, l2_alpha=0.01)
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.5, 2.0, 10.0])  # Large outlier at 10.0
    >>> loss = huber(y_true, y_pred)
    
    Notes
    -----
    - delta=1.0 is a common default; tune based on your data scale
    - More robust to outliers than MSE
    - Differentiable everywhere (unlike MAE at zero)
    """
    
    def __init__(self, delta=1.0, l2_alpha=0.0):
        self.delta = delta
        self.l2_alpha = l2_alpha

    def __call__(self, y_true, y_pred, weights=None):
        """
        Compute Huber loss with optional L2 regularization.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_pred : np.ndarray
            Predicted values.
        weights : list of np.ndarray, optional
            Model weights for L2 regularization.
            
        Returns
        -------
        float
            The total loss (Huber + L2 regularization if applicable).
        """
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        huber_loss = np.where(
            is_small_error,
            0.5 * error**2,
            self.delta * (np.abs(error) - 0.5 * self.delta)
        )
        huber_loss = np.mean(huber_loss)
        l2 = self.l2_regularization(weights, y_true.shape[0])
        return huber_loss + l2

    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of Huber loss with respect to the output logits.

        This implementation assumes a linear output activation.
        """
        if output_activation_name != "linear":
            raise NotImplementedError(
                "Gradient for Huber loss is only implemented for linear output activation."
            )

        error = y_pred - y_true
        grad = np.where(
            np.abs(error) <= self.delta,
            error,  
            self.delta * np.sign(error)
        )
        return grad

class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy loss function.
    
    Used for multi-class classification where labels are one-hot encoded.
    Measures the dissimilarity between the true distribution and predicted probabilities.
    
    Formula
    -------
    CCE = -(1/N) * sum(sum(y_true * log(y_pred)))
    
    With L2 regularization:
    Loss = CCE + (l2_alpha / (2N)) * sum(w^2)
    
    Parameters
    ----------
    l2_alpha : float, default=0.0
        L2 regularization strength.
        
    Methods
    -------
    __call__(y_true, y_pred, weights=None)
        Compute categorical cross-entropy with optional L2 regularization.
        
    Examples
    --------
    >>> cce = CategoricalCrossEntropy()
    >>> y_true = np.array([[1, 0, 0], [0, 1, 0]])  # One-hot encoded
    >>> y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    >>> loss = cce(y_true, y_pred)
    
    Notes
    -----
    - Requires softmax activation in the output layer
    - y_pred values are clipped to [epsilon, 1-epsilon] for numerical stability
    - y_true must be one-hot encoded (not integer labels)
    """
    
    def __init__(self, l2_alpha=0.0):
        self.l2_alpha = l2_alpha

    def __call__(self, y_true, y_pred, weights=None):
        """
        Compute categorical cross-entropy with optional L2 regularization.
        
        Parameters
        ----------
        y_true : np.ndarray
            One-hot encoded true labels, shape (n_samples, n_classes).
        y_pred : np.ndarray
            Predicted probabilities from softmax, shape (n_samples, n_classes).
            Values are automatically clipped to avoid log(0).
        weights : list of np.ndarray, optional
            Model weights for L2 regularization.
            
        Returns
        -------
        float
            The total loss (CCE + L2 regularization if applicable).
            
        Notes
        -----
        - Automatically clips predictions to [1e-15, 1-1e-15] for numerical stability
        - Each row of y_pred should sum to approximately 1.0 (softmax output)
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        ce = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        l2 = self.l2_regularization(weights, y_true.shape[0])
        return ce + l2

    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of CCE with respect to the output logits.

        This implementation assumes a softmax output activation.
        """
        if output_activation_name != "softmax":
            raise NotImplementedError(
                "Gradient for CCE is only implemented for softmax output activation."
            )

        
        return y_pred - y_true

class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy (BCE) loss function.
    
    Used for binary classification or multi-label classification.
    Measures the dissimilarity between true binary labels and predicted probabilities.
    
    Formula
    -------
    BCE = -(1/N) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    
    With L2 regularization:
    Loss = BCE + (l2_alpha / (2N)) * sum(w^2)
    
    Parameters
    ----------
    l2_alpha : float, default=0.0
        L2 regularization strength.
        
    Methods
    -------
    __call__(y_true, y_pred, weights=None)
        Compute binary cross-entropy with optional L2 regularization.
        
    Examples
    --------
    >>> bce = BinaryCrossEntropy()
    >>> y_true = np.array([1, 0, 1, 0])
    >>> y_pred = np.array([0.9, 0.1, 0.8, 0.3])
    >>> loss = bce(y_true, y_pred)
    
    Notes
    -----
    - Requires sigmoid activation in the output layer
    - y_true should be 0 or 1 (or probabilities for soft labels)
    - For multi-label: each output is treated independently
    """
    
    def __init__(self, l2_alpha=0.0):
        self.l2_alpha = l2_alpha

    def __call__(self, y_true, y_pred, weights=None):
        """
        Compute binary cross-entropy with optional L2 regularization.
        
        Parameters
        ----------
        y_true : np.ndarray
            True binary labels (0 or 1), shape (n_samples,) or (n_samples, n_outputs).
        y_pred : np.ndarray
            Predicted probabilities from sigmoid, same shape as y_true.
            Values are automatically clipped to avoid log(0).
        weights : list of np.ndarray, optional
            Model weights for L2 regularization.
            
        Returns
        -------
        float
            The total loss (BCE + L2 regularization if applicable).
            
        Notes
        -----
        - Automatically clips predictions to [1e-15, 1-1e-15] for numerical stability
        - Each element is treated independently (no sum over axis=1 unlike CCE)
        """
        epsilon = 1e-15
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
    
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        l2 = self.l2_regularization(weights, y_true.shape[0])
        return bce + l2

    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of BCE with respect to the output logits.

        This implementation assumes a sigmoid output activation.
        """
        if output_activation_name != "sigmoid":
            raise NotImplementedError(
                "Gradient for BCE is only implemented for sigmoid output activation."
            )

        return y_pred - y_true

class LogCoshLoss(Loss):
    """
    Log-Cosh loss function.
    
    Computes the logarithm of the hyperbolic cosine of the prediction error.
    Similar to MSE but less sensitive to outliers.
    
    Formula
    -------
    LogCosh = (1/N) * sum(log(cosh(y_pred - y_true)))
    
    With L2 regularization:
    Loss = LogCosh + (l2_alpha / (2N)) * sum(w^2)
    
    Parameters
    ----------
    l2_alpha : float, default=0.0
        L2 regularization strength.
        
    Methods
    -------
    __call__(y_true, y_pred, weights=None)
        Compute log-cosh loss with optional L2 regularization.
        
    Examples
    --------
    >>> logcosh = LogCoshLoss()
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.2, 1.9, 3.1])
    >>> loss = logcosh(y_true, y_pred)
    
    Notes
    -----
    - Approximates MSE for small errors, MAE for large errors
    - Twice differentiable everywhere (unlike Huber)
    - log(cosh(x)) ≈ x^2/2 for small x, ≈ |x| - log(2) for large x
    """
    
    def __init__(self, l2_alpha=0.0):
        self.l2_alpha = l2_alpha

    def __call__(self, y_true, y_pred, weights=None):
        """
        Compute Log-Cosh loss with optional L2 regularization.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values.
        y_pred : np.ndarray
            Predicted values.
        weights : list of np.ndarray, optional
            Model weights for L2 regularization.
            
        Returns
        -------
        float
            The total loss (LogCosh + L2 regularization if applicable).
        """
        log_cosh = np.mean(np.log(np.cosh(y_pred - y_true)))
        l2 = self.l2_regularization(weights, y_true.shape[0])
        return log_cosh + l2
    
    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of Log-Cosh loss with respect to the output logits.

        This implementation assumes a linear output activation.
        """
        if output_activation_name != "linear":
            raise NotImplementedError(
                "Gradient for Log-Cosh loss is only implemented for linear output activation."
            )

        return np.tanh(y_pred - y_true)

class MeanSquaredLogarithmicError(Loss):
    """
    Mean Squared Logarithmic Error (MSLE) loss function.
    
    Computes the mean squared error of the logarithm of predictions.
    Useful when targets have large values and you care about relative errors.
    
    Formula
    -------
    MSLE = (1/N) * sum((log(1 + y_pred) - log(1 + y_true))^2)
    
    With L2 regularization:
    Loss = MSLE + (l2_alpha / (2N)) * sum(w^2)
    
    Parameters
    ----------
    l2_alpha : float, default=0.0
        L2 regularization strength.
        
    Methods
    -------
    __call__(y_true, y_pred, weights=None)
        Compute MSLE with optional L2 regularization.
        
    Examples
    --------
    >>> msle = MeanSquaredLogarithmicError()
    >>> y_true = np.array([100, 1000, 10000])
    >>> y_pred = np.array([110, 950, 10500])
    >>> loss = msle(y_true, y_pred)
    
    Notes
    -----
    - Penalizes relative errors (good for targets with large ranges)
    - Only positive values allowed (clipped to epsilon if needed)
    - Useful for house prices, population counts, etc.
    - log1p(x) = log(1+x) used for numerical stability
    """
    
    def __init__(self, l2_alpha=0.0):
        self.l2_alpha = l2_alpha

    def __call__(self, y_true, y_pred, weights=None):
        """
        Compute MSLE with optional L2 regularization.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth target values (must be non-negative).
        y_pred : np.ndarray
            Predicted values (must be non-negative, clipped to epsilon if needed).
        weights : list of np.ndarray, optional
            Model weights for L2 regularization.
            
        Returns
        -------
        float
            The total loss (MSLE + L2 regularization if applicable).
            
        Notes
        -----
        - Negative values in y_true or y_pred are clipped to epsilon (1e-15)
        - Uses log1p for numerical stability: log1p(x) = log(1+x)
        """
        epsilon = 1e-15
        y_true = np.clip(y_true, epsilon, None)
        y_pred = np.clip(y_pred, epsilon, None)
        msle = np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true)))
        l2 = self.l2_regularization(weights, y_true.shape[0])
        return msle + l2

    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of MSLE with respect to the output logits.

        This implementation assumes a linear output activation.
        """
        if output_activation_name != "linear":
            raise NotImplementedError(
                "Gradient for MSLE is only implemented for linear output activation."
            )

        epsilon = 1e-15
        y_t = np.clip(y_true, epsilon, None)
        y_p = np.clip(y_pred, epsilon, None)
        
        return 2.0 * (np.log1p(y_p) - np.log1p(y_t)) / (1.0 + y_p)

class BinaryFocalLoss(Loss):
    """
    Binary Focal Loss function.
    
    Extension of binary cross-entropy that down-weights easy examples.
    Useful for highly imbalanced datasets (e.g., object detection).
    
    Formula
    -------
    FL = -(1/N) * sum(alpha * (1 - y_pred)^gamma * y_true * log(y_pred) +
                      (1 - alpha) * y_pred^gamma * (1 - y_true) * log(1 - y_pred))
    
    With L2 regularization:
    Loss = FL + (l2_alpha / (2N)) * sum(w^2)
    
    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter. Higher gamma increases focus on hard examples.
        - gamma=0: equivalent to weighted BCE
        - gamma=2: standard focal loss (recommended)
    alpha : float, default=0.25
        Balance parameter for positive/negative classes.
        - alpha=0.25: down-weights positive class (standard for detection)
        - alpha=0.5: balanced
        - alpha=0.75: up-weights positive class
    l2_alpha : float, default=0.0
        L2 regularization strength.
        
    Methods
    -------
    __call__(y_true, y_pred, weights=None)
        Compute focal loss with optional L2 regularization.
        
    Examples
    --------
    >>> focal = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    >>> y_true = np.array([1, 0, 1, 0])
    >>> y_pred = np.array([0.9, 0.1, 0.8, 0.3])
    >>> loss = focal(y_true, y_pred)
    
    Notes
    -----
    - From "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    - Best for highly imbalanced datasets
    - Requires sigmoid activation
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, l2_alpha=0.0):
        self.gamma = gamma
        self.alpha = alpha
        self.l2_alpha = l2_alpha

    def __call__(self, y_true, y_pred, weights=None):
        """
        Compute binary focal loss with optional L2 regularization.
        
        Parameters
        ----------
        y_true : np.ndarray
            True binary labels (0 or 1).
        y_pred : np.ndarray
            Predicted probabilities from sigmoid.
            Values are automatically clipped to avoid log(0).
        weights : list of np.ndarray, optional
            Model weights for L2 regularization.
            
        Returns
        -------
        float
            The total loss (Focal + L2 regularization if applicable).
        """
        
        epsilon = 1e-15
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
        
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        focal_loss = -np.mean(np.sum(
            self.alpha * (1 - y_pred) ** self.gamma * y_true * np.log(y_pred) + 
            (1 - self.alpha) * y_pred ** self.gamma * (1 - y_true) * np.log(1 - y_pred),
            axis=1
        ))
        l2 = self.l2_regularization(weights, y_true.shape[0])
        return focal_loss + l2

    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of binary focal loss with respect to the output logits.

        This implementation assumes a sigmoid output activation.
        """
        if output_activation_name != "sigmoid":
            raise NotImplementedError(
                "Gradient for Binary Focal Loss is only implemented for sigmoid output activation."
            )

        epsilon = 1e-15
        p = np.clip(y_pred, epsilon, 1 - epsilon)
        y = y_true
        gamma = self.gamma
        alpha = self.alpha
        term_pos = -alpha * (-gamma * (1 - p)**(gamma - 1) * np.log(p) + (1 - p)**gamma / p)
        term_neg = -(1 - alpha) * (gamma * p**(gamma - 1) * np.log(1 - p) - p**gamma / (1 - p))

        dL_dp = term_pos * y + term_neg * (1 - y)
        dp_dz = p * (1 - p)  # Derivative of sigmoid
        grad = dL_dp * dp_dz
        
        return grad

class KLDivLoss(Loss):
    """
    Kullback-Leibler Divergence loss function.
    
    Measures how one probability distribution diverges from a second,
    expected probability distribution. Asymmetric: KL(P||Q) != KL(Q||P).
    
    Formula
    -------
    KL = (1/N) * sum(sum(y_true * log(y_true / y_pred)))
    
    With L2 regularization:
    Loss = KL + (l2_alpha / (2N)) * sum(w^2)
    
    Parameters
    ----------
    l2_alpha : float, default=0.0
        L2 regularization strength.
        
    Methods
    -------
    __call__(y_true, y_pred, weights=None)
        Compute KL divergence with optional L2 regularization.
        
    Examples
    --------
    >>> kl = KLDivLoss()
    >>> y_true = np.array([[0.3, 0.7], [0.8, 0.2]])  # Must sum to 1
    >>> y_pred = np.array([[0.4, 0.6], [0.7, 0.3]])
    >>> loss = kl(y_true, y_pred)
    
    Notes
    -----
    - y_true MUST be a valid probability distribution (sums to 1)
    - y_pred should also be a probability distribution (softmax output)
    - Asymmetric: KL(true||pred) is used here (forward KL)
    - Common in VAEs, distillation, and variational inference
    - Returns 0 when y_true == y_pred

    WARNING 
    -------
    - KL divergence is not a true metric (not symmetric, does not satisfy triangle inequality)
    - Use with care and understand the implications for your specific application.
    """
    
    def __init__(self, l2_alpha=0.0):
        self.l2_alpha = l2_alpha

    def __call__(self, y_true, y_pred, weights=None):
        """
        Compute KL divergence with optional L2 regularization.
        
        Parameters
        ----------
        y_true : np.ndarray
            True probability distribution. Must sum to 1 along axis=1.
        y_pred : np.ndarray
            Predicted probability distribution. Should sum to ~1 (softmax output).
            Values are automatically clipped for numerical stability.
        weights : list of np.ndarray, optional
            Model weights for L2 regularization.
            
        Returns
        -------
        float
            The total loss (KL + L2 regularization if applicable).
            
        Notes
        -----
        - Automatically clips values to [1e-15, 1] for numerical stability
        - Use forward KL: KL(true || pred)
        """
        epsilon = 1e-15
        y_true = np.clip(y_true, epsilon, 1)
        y_pred = np.clip(y_pred, epsilon, 1)
        kl_div_loss = np.mean(np.sum(y_true * np.log(y_true / y_pred), axis=1))
        l2 = self.l2_regularization(weights, y_true.shape[0])
        return kl_div_loss + l2
    
    def gradient(self, y_true, y_pred, output_activation_name):
        """
        Compute the gradient of KL divergence with respect to the output logits.

        This implementation assumes a softmax output activation.
        """
        if output_activation_name != "softmax":
            raise NotImplementedError(
                "Gradient for KL Divergence is only implemented for softmax output activation."
            )

        epsilon = 1e-15
        y_true = np.clip(y_true, epsilon, 1)
        y_pred = np.clip(y_pred, epsilon, 1)

        return y_pred - y_true


# Dictionary to map task types to corresponding loss functions
LOSS_FUNCTIONS = {
    "binary_cross_entropy": BinaryCrossEntropy,
    "categorical_cross_entropy": CategoricalCrossEntropy,
    "mean_squared_error": MeanSquaredError,
    "mean_absolute_error": MeanAbsoluteError,
    "huber_loss": HuberLoss,
    "log_cosh_loss": LogCoshLoss,
    "mean_squared_log_error": MeanSquaredLogarithmicError,
    "binary_focal_loss": BinaryFocalLoss,
    "kl_divergence": KLDivLoss
}


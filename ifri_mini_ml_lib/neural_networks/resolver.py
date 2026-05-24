from typing import Dict, Any
from .loss import LOSS_FUNCTIONS

# Configuration for allowed losses based on task and output activation
TASK_CONFIG = {
    "classification": {
        "outputs": {"sigmoid", "softmax"},
        "default_output": "softmax",
        "default_loss": {
            "sigmoid": "binary_cross_entropy",
            "softmax": "categorical_cross_entropy"
        },
        "losses": {
            "sigmoid": {"binary_cross_entropy", "binary_focal_loss"},
            "softmax": {"categorical_cross_entropy", "kl_divergence"} # CAUTION: KL divergence is not a true loss function for classification, but can be used in certain contexts (e.g., teacher-student training)
        }
    },

    "regression": {
        "outputs": {"linear"},
        "default_output": "linear",
        "default_loss": {
            "linear": "mean_squared_error"
        },
        "losses": {
            "linear": {
                "mean_squared_error",
                "mean_absolute_error",
                "huber_loss",
                "log_cosh_loss",
                "mean_squared_log_error"
            }
        }
    }
}


def resolve_config(task, output_activation, loss_name):
    """
    Resolve the output activation and loss function based on the task and user input.
    
    Parameters
    ----------
    task : str
        The task type. Must be either "classification" or "regression".
    output_activation : str or None
        The output activation function. If "auto" or None, it will be set to the default for the task.
    loss_name : str or None
        The name of the loss function. If "auto" or None, it will be set to the default for the task and output activation.
    
    Returns
    -------
    output_activation : str
        The resolved output activation function.
    loss_name : str
        The resolved loss function name.
    """
    config = TASK_CONFIG[task]

    # OUTPUT
    if output_activation == "auto" or output_activation is None:
        output_activation = config["default_output"]

    if output_activation not in config["outputs"]:
        raise ValueError(
            f"Invalid output activation '{output_activation}' for task '{task}'. "
            f"Allowed outputs: {config['outputs']}"
        )
    
    # LOSS
    if loss_name is None or loss_name == "auto":
        loss_name = config["default_loss"][output_activation]

    return output_activation, loss_name

def build_loss_kwargs(loss_name: str, loss_params:Dict[str, Any], alpha: float ) -> Dict[str, Any]:
    """
    Builds the kwargs for instantiating the loss function.
    Merges default parameters, user-provided parameters, and l2_alpha.
    """
    # Default parameters for each loss
    defaults = {
        "binary_cross_entropy": {"l2_alpha": 0.0},
        "categorical_cross_entropy": {"l2_alpha": 0.0},
        "mean_squared_error": {"l2_alpha": 0.0},
        "mean_absolute_error": {"l2_alpha": 0.0},
        "huber_loss": {"delta": 1.0, "l2_alpha": 0.0},
        "log_cosh_loss": {"l2_alpha": 0.0},
        "mean_squared_log_error": {"l2_alpha": 0.0},
        "binary_focal_loss": {"gamma": 2.0, "alpha": 0.25, "l2_alpha": 0.0},
        "kl_divergence": {"l2_alpha": 0.0},
    }
    
    # Start with default values
    kwargs = defaults.get(loss_name, {}).copy()
    
    # Update with user-provided parameters (overrides defaults)
    kwargs.update(loss_params)
    
    # Ensure l2_alpha is set to alpha if not provided by user
    if "l2_alpha" not in loss_params:
        kwargs["l2_alpha"] = alpha
        
    return kwargs

# Validation function to check if the chosen loss is compatible with the task and output activation
def validate_config(task, output_activation, loss_name):
    """
    Validate that a loss function is compatible with the task and output activation.
    
    Parameters
    ----------
    task : str
        The task type. Must be either "classification" or "regression".
    output_activation : str
        The output activation function. Must be in TASK_CONFIG[task]["outputs"].
    loss_name : str
        The name of the loss function. Must be in LOSS_FUNCTIONS keys.
        
    Returns
    -------
    None
        Returns silently if configuration is valid.
        
    Raises
    ------
    ValueError
        If output_activation is not allowed for the task, or if loss_name is
        incompatible with the output_activation.
        
    Examples
    --------
    >>> validate_config("classification", "sigmoid", "binary_cross_entropy")
    >>> validate_config("regression", "linear", "mean_squared_error")
    >>> validate_config("classification", "softmax", "binary_cross_entropy")
    ValueError: Loss 'binary_cross_entropy' incompatible with output 'softmax'
    """
    config = TASK_CONFIG[task]

    if output_activation not in config["outputs"]:
        raise ValueError(
            f"Invalid output '{output_activation}' for task '{task}'"
        )

    if loss_name not in config["losses"][output_activation]:
        raise ValueError(
            f"Loss '{loss_name}' not compatible with output '{output_activation}'"
        )
    
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Loss '{loss_name}' is not implemented. Available losses: {list(LOSS_FUNCTIONS.keys())}"
        )
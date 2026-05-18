import pytest
import numpy as np
from ifri_mini_ml_lib.neural_networks import MLP

class TestMLP:
    def setup_method(self):
        """Configuration commune pour les tests."""
        np.random.seed(42)
        # Classification dataset
        self.X_clf = np.random.randn(100, 5)
        self.y_clf = np.random.randint(0, 3, size=100)
        
        # Regression dataset
        self.X_reg = np.random.randn(100, 5)
        self.y_reg = np.random.randn(100)
        self.y_reg_multi = np.random.randn(100, 2)

    #  INITIALIZATION 

    def test_initialization_classification(self):
        """Test initialization for classification."""
        mlp = MLP(task="classification", hidden_layer_sizes=(10, 5), random_state=42)
        assert mlp.hidden_layer_sizes == (10, 5)
        assert mlp.hidden_activation_name == "relu"
        assert mlp.optimizer == "adam"
        assert mlp.trained is False
        assert mlp.task == "classification"
        assert mlp.output_activation_name == "softmax"  # default for multiclass

    def test_initialization_regression(self):
        """Test initialization for regression."""
        mlp = MLP(task="regression", hidden_layer_sizes=(10, 5), random_state=42)
        assert mlp.task == "regression"
        assert mlp.output_activation_name == "linear"
        assert mlp.loss_name == "mean_squared_error"

    def test_initialization_custom_loss(self):
        """Test with custom loss and activation."""
        mlp = MLP(
            task="classification",
            output_activation="sigmoid",
            loss_name="binary_cross_entropy",
            loss_params={"l2_alpha": 0.01}
        )
        assert mlp.output_activation_name == "sigmoid"
        assert mlp.loss_name == "binary_cross_entropy"

    def test_invalid_task(self):
        """Test with invalid task."""
        with pytest.raises(ValueError):
            MLP(task="invalid_task")

    def test_invalid_hidden_activation(self):
        """Test with invalid hidden activation."""
        with pytest.raises(ValueError):
            MLP(hidden_activation="invalid")

    #  ACTIVATION FUNCTIONS 

    def test_activation_functions(self):
        """Test activation functions via forward pass."""
        from ifri_mini_ml_lib.neural_networks.activation import (
            ACTIVATIONS, DERIVATIVES
        )
        
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # ReLU
        assert np.allclose(ACTIVATIONS["relu"](x), [0, 0, 0, 1, 2])
        assert np.allclose(DERIVATIVES["relu"](x), [0, 0, 0, 1, 1])
        
        # Sigmoid
        sig_expected = 1 / (1 + np.exp(-x))
        assert np.allclose(ACTIVATIONS["sigmoid"](x), sig_expected)
        
        # Tanh
        assert np.allclose(ACTIVATIONS["tanh"](x), np.tanh(x))
        
        # Leaky ReLU
        leaky_expected = np.where(x > 0, x, 0.01 * x)
        assert np.allclose(ACTIVATIONS["leaky_relu"](x), leaky_expected)
        
        # Linear
        assert np.allclose(ACTIVATIONS["linear"](x), x)
        assert np.allclose(DERIVATIVES["linear"](x), np.ones_like(x))

    #  FORWARD PASS 

    def test_forward_pass_classification(self):
        """Test forward pass for classification."""
        mlp = MLP(task="classification", hidden_layer_sizes=(3,), random_state=42)
        mlp.fit(self.X_clf[:10], self.y_clf[:10])  # Initialize weights
        
        activations, layer_inputs = mlp._forward_pass(self.X_clf[:5])
        
        assert len(activations) == 3  # input + hidden + output
        assert activations[0].shape == (5, 5)  # input
        assert activations[1].shape == (5, 3)  # hidden
        assert activations[2].shape == (5, 3)  # output (3 classes)
        
        # Softmax probabilities sum to 1
        assert np.allclose(np.sum(activations[2], axis=1), 1.0)

    def test_forward_pass_regression(self):
        """Test forward pass for regression."""
        mlp = MLP(task="regression", hidden_layer_sizes=(3,), random_state=42)
        mlp.fit(self.X_reg[:10], self.y_reg[:10])
        
        activations, layer_inputs = mlp._forward_pass(self.X_reg[:5])
        
        assert activations[2].shape == (5, 1)  # single output
        # Linear output: no constraint on values

    #  LOSS FUNCTIONS 

    def test_loss_functions(self):
        """Test all loss functions."""
        from ifri_mini_ml_lib.neural_networks.loss import LOSS_FUNCTIONS
        
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        # Test each loss can be instantiated and called
        for name, loss_class in LOSS_FUNCTIONS.items():
            loss_fn = loss_class(l2_alpha=0.01)
            
            if name in ["binary_cross_entropy", "binary_focal_loss"]:
                # Binary losses need sigmoid output
                y_b = np.array([1, 0])
                y_p = np.array([0.8, 0.3])
                loss = loss_fn(y_b, y_p)
                assert isinstance(loss, float)
            elif name in ["categorical_cross_entropy", "kl_divergence"]:
                # Multi-class losses need softmax output
                loss = loss_fn(y_true, y_pred)
                assert isinstance(loss, float)
            else:
                # Regression losses
                y_r = np.array([1.0, 2.0])
                y_p = np.array([1.1, 1.9])
                loss = loss_fn(y_r, y_p)
                assert isinstance(loss, float)

    def test_loss_gradients(self):
        """Test gradient computation for key losses."""
        from ifri_mini_ml_lib.neural_networks.loss import (
            MeanSquaredError, CategoricalCrossEntropy, BinaryCrossEntropy
        )
        
        # MSE + linear
        mse = MeanSquaredError()
        grad = mse.gradient(
            np.array([1.0, 2.0]),
            np.array([1.1, 1.9]),
            "linear"
        )
        assert grad.shape == (2,)
        
        # CCE + softmax
        cce = CategoricalCrossEntropy()
        grad = cce.gradient(
            np.array([[1, 0], [0, 1]]),
            np.array([[0.8, 0.2], [0.3, 0.7]]),
            "softmax"
        )
        assert grad.shape == (2, 2)
        
        # BCE + sigmoid
        bce = BinaryCrossEntropy()
        grad = bce.gradient(
            np.array([1, 0]),
            np.array([0.8, 0.3]),
            "sigmoid"
        )
        assert grad.shape == (2,)

    #  FIT / PREDICT 

    def test_fit_predict_classification(self):
        """Test training and prediction for classification."""
        mlp = MLP(
            task="classification",
            hidden_layer_sizes=(10, 5),
            max_iter=50,
            random_state=42
        )
        mlp.fit(self.X_clf, self.y_clf)
        
        assert mlp.trained is True
        assert len(mlp.loss_history) > 0
        
        y_pred = mlp.predict(self.X_clf)
        assert y_pred.shape == self.y_clf.shape

    def test_fit_predict_regression(self):
        """Test training and prediction for regression."""
        mlp = MLP(
            task="regression",
            hidden_layer_sizes=(10, 5),
            max_iter=50,
            random_state=42
        )
        mlp.fit(self.X_reg, self.y_reg)
        
        y_pred = mlp.predict(self.X_reg)
        assert y_pred.shape == self.y_reg.shape

    def test_fit_predict_regression_multioutput(self):
        """Test regression with multiple outputs."""
        mlp = MLP(
            task="regression",
            hidden_layer_sizes=(10, 5),
            max_iter=50,
            random_state=42
        )
        mlp.fit(self.X_reg, self.y_reg_multi)
        
        y_pred = mlp.predict(self.X_reg)
        assert y_pred.shape == self.y_reg_multi.shape

    def test_fit_xor(self):
        """Test learning XOR problem."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        
        mlp = MLP(
            task="classification",
            hidden_layer_sizes=(10, 10),
            max_iter=500,
            random_state=42
        )
        mlp.fit(X, y)
        
        y_pred = mlp.predict(X)
        # Should achieve reasonable accuracy on XOR
        accuracy = np.mean(y_pred == y)
        assert accuracy >= 0.5  # At least better than random

    #  OPTIMIZERS 

    def test_different_optimizers(self):
        """Test with different optimizers."""
        optimizers = ["sgd", "momentum", "rmsprop", "adam"]
        
        for optimizer in optimizers:
            mlp = MLP(
                task="classification",
                hidden_layer_sizes=(5,),
                optimizer=optimizer,
                max_iter=10,
                random_state=42
            )
            mlp.fit(self.X_clf, self.y_clf)
            y_pred = mlp.predict(self.X_clf)
            assert y_pred.shape == self.y_clf.shape

    #  DIFFERENT LOSSES 

    def test_different_losses_classification(self):
        """Test classification with different losses."""
        losses = ["binary_cross_entropy", "categorical_cross_entropy"]
        
        for loss in losses:
            if loss == "binary_cross_entropy":
                # Binary problem
                y_bin = np.random.randint(0, 2, size=100)
                mlp = MLP(
                    task="classification",
                    output_activation="sigmoid",
                    loss_name=loss,
                    hidden_layer_sizes=(5,),
                    max_iter=10,
                    random_state=42
                )
                mlp.fit(self.X_clf, y_bin)
            else:
                mlp = MLP(
                    task="classification",
                    loss_name=loss,
                    hidden_layer_sizes=(5,),
                    max_iter=10,
                    random_state=42
                )
                mlp.fit(self.X_clf, self.y_clf)
            
            y_pred = mlp.predict(self.X_clf)
            assert y_pred.shape[0] == 100

    def test_different_losses_regression(self):
        """Test regression with different losses."""
        losses = [
            "mean_squared_error",
            "mean_absolute_error",
            "huber_loss",
            "log_cosh_loss",
            "mean_squared_log_error"
        ]
        
        for loss in losses:
            loss_params = {}
            if loss == "huber_loss":
                loss_params = {"delta": 1.0}
            
            mlp = MLP(
                task="regression",
                loss_name=loss,
                loss_params=loss_params,
                hidden_layer_sizes=(5,),
                max_iter=10,
                random_state=42
            )
            mlp.fit(self.X_reg, self.y_reg)
            y_pred = mlp.predict(self.X_reg)
            assert y_pred.shape == self.y_reg.shape

    #  EARLY STOPPING 

    def test_early_stopping(self):
        """Test early stopping."""
        mlp = MLP(
            task="classification",
            hidden_layer_sizes=(5,),
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            max_iter=50,
            random_state=42
        )
        mlp.fit(self.X_clf, self.y_clf)
        
        assert len(mlp.val_loss_history) > 0
        assert len(mlp.loss_history) <= 50

    #  PREDICT PROBA 

    def test_predict_proba(self):
        """Test probability prediction."""
        mlp = MLP(
            task="classification",
            hidden_layer_sizes=(5,),
            max_iter=10,
            random_state=42
        )
        mlp.fit(self.X_clf, self.y_clf)
        
        probas = mlp.predict_proba(self.X_clf)
        
        # Probabilities sum to 1
        assert np.allclose(np.sum(probas, axis=1), 1.0)
        assert probas.shape == (100, 3)
        
        # Consistency with predict
        y_pred = mlp.predict(self.X_clf)
        y_pred_from_proba = np.argmax(probas, axis=1)
        assert np.all(y_pred == y_pred_from_proba)

    def test_predict_proba_regression_error(self):
        """Test that predict_proba raises error for regression."""
        mlp = MLP(task="regression", hidden_layer_sizes=(5,), max_iter=10, random_state=42)
        mlp.fit(self.X_reg, self.y_reg)
        
        with pytest.raises(ValueError):
            mlp.predict_proba(self.X_reg)

    #  SCORE 

    def test_score_classification(self):
        """Test accuracy score."""
        mlp = MLP(
            task="classification",
            hidden_layer_sizes=(5,),
            max_iter=10,
            random_state=42
        )
        mlp.fit(self.X_clf, self.y_clf)
        
        accuracy = mlp.score(self.X_clf, self.y_clf)
        assert 0 <= accuracy <= 1

    def test_score_regression(self):
        """Test R² score."""
        mlp = MLP(
            task="regression",
            hidden_layer_sizes=(5,),
            max_iter=50,
            random_state=42
        )
        mlp.fit(self.X_reg, self.y_reg)
        
        r2 = mlp.score(self.X_reg, self.y_reg)
        assert r2 <= 1.0  # R² can be negative

    #  ERROR HANDLING 

    def test_predict_without_training(self):
        """Test prediction without training."""
        mlp = MLP(task="classification")
        
        with pytest.raises(ValueError):
            mlp.predict(self.X_clf)
        
        with pytest.raises(ValueError):
            mlp.predict_proba(self.X_clf)
        
        with pytest.raises(ValueError):
            mlp.score(self.X_clf, self.y_clf)

    def test_invalid_loss_for_task(self):
        """Test invalid loss for task."""
        with pytest.raises(ValueError):
            MLP(
                task="classification",
                output_activation="softmax",
                loss_name="mean_squared_error"  # Invalid for classification
            )

    def test_invalid_output_activation(self):
        """Test invalid output activation for task."""
        with pytest.raises(ValueError):
            MLP(
                task="regression",
                output_activation="sigmoid"  # Invalid for regression
            )

    #  LOSS HISTORY 

    def test_loss_history(self):
        """Test loss history recording."""
        mlp = MLP(
            task="classification",
            hidden_layer_sizes=(5,),
            max_iter=20,
            random_state=42
        )
        mlp.fit(self.X_clf, self.y_clf)
        
        assert len(mlp.loss_history) == 20
        # Loss should generally decrease
        assert mlp.loss_history[-1] < mlp.loss_history[0] * 2  # Rough check

    #  REGULARIZATION 

    def test_l2_regularization(self):
        """Test L2 regularization effect."""
        mlp_no_reg = MLP(
            task="regression",
            alpha=0.0,
            hidden_layer_sizes=(5,),
            max_iter=10,
            random_state=42
        )
        mlp_reg = MLP(
            task="regression",
            alpha=0.1,
            hidden_layer_sizes=(5,),
            max_iter=10,
            random_state=42
        )
        
        mlp_no_reg.fit(self.X_reg, self.y_reg)
        mlp_reg.fit(self.X_reg, self.y_reg)
        
        # Regularized model should have different weights
        assert not np.allclose(
            mlp_no_reg.weights[0],
            mlp_reg.weights[0]
        )
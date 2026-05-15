import pytest
import numpy as np
from ifri_mini_ml_lib.classification.svm import (
    SVMLinear, SVMRBF, SVMRBFOvO, BaseSVM, 
    rbf_kernel, smo, LinearSolver
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures: Common test data
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_data_simple():
    """Simple linearly separable binary data."""
    X = np.array([
        [1, 1],
        [2, 2],
        [2, 0],
        [0, 0],
        [0, 1],
        [1, 0],
    ], dtype=float)
    y = np.array([-1, -1, -1, 1, 1, 1])
    return X, y


@pytest.fixture
def binary_data_normalized():
    """Linearly separable binary data (normalized)."""
    X = np.array([
        [-1, -1],
        [-2, -1],
        [1, 1],
        [2, 1],
    ], dtype=float)
    y = np.array([-1, -1, 1, 1])
    return X, y


@pytest.fixture
def multiclass_data():
    """Multiclass data with 3 classes."""
    np.random.seed(42)
    X = np.array([
        [0, 0], [0, 1], [1, 0],      # Class 0
        [3, 3], [3, 4], [4, 3],      # Class 1
        [6, 6], [6, 7], [7, 6],      # Class 2
    ], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return X, y


@pytest.fixture
def multiclass_data_labels():
    """Multiclass data with string labels."""
    X = np.array([
        [0, 0], [0, 1],
        [3, 3], [3, 4],
        [6, 6], [6, 7],
    ], dtype=float)
    y = np.array(['cat', 'cat', 'dog', 'dog', 'bird', 'bird'])
    return X, y


# ─────────────────────────────────────────────────────────────────────
# Tests: BaseSVM
# ─────────────────────────────────────────────────────────────────────

def test_base_svm_cannot_instantiate():
    """BaseSVM is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseSVM()


def test_base_svm_check_is_fitted():
    """Test the _check_is_fitted method."""
    svm = SVMLinear(C=1.0)
    
    # Should raise AttributeError before fitting
    with pytest.raises(AttributeError, match="not yet trained"):
        svm.predict([[1, 2]])


# ─────────────────────────────────────────────────────────────────────
# Tests: LinearSolver (Pegasos)
# ─────────────────────────────────────────────────────────────────────

def test_linear_solver_initialization():
    """Test LinearSolver initialization."""
    solver = LinearSolver(C=2.0, max_iter=500, tol=1e-5)
    assert solver.C == 2.0
    assert solver.max_iter == 500
    assert solver.tol == 1e-5


def test_linear_solver_solve(binary_data_simple):
    """Test Pegasos optimization."""
    X, y = binary_data_simple
    solver = LinearSolver(C=1.0, max_iter=100, random_state=42)
    w, b = solver.solve(X, y)
    
    # Check shapes
    assert w.shape == (2,)
    assert isinstance(b, float)
    
    # Check loss history
    assert len(solver.loss_history_) > 0
    assert solver.loss_history_[-1] < solver.loss_history_[0]  # Loss decreases


def test_linear_solver_convergence(binary_data_simple):
    """Test that solver converges."""
    X, y = binary_data_simple
    solver = LinearSolver(C=1.0, max_iter=1000, tol=1e-3, random_state=42)
    w, b = solver.solve(X, y)
    
    # For simple data, loss should be relatively small
    final_loss = solver.loss_history_[-1]
    assert final_loss < 1.0


# ─────────────────────────────────────────────────────────────────────
# Tests: SVMLinear
# ─────────────────────────────────────────────────────────────────────

def test_svm_linear_initialization():
    """Test SVMLinear initialization."""
    svm = SVMLinear(C=2.0, n_iters=500, tol=1e-5, random_state=42)
    assert svm.C == 2.0
    assert svm.n_iters == 500
    assert svm.tol == 1e-5
    assert svm.random_state == 42
    assert svm.is_fitted_ is False


def test_svm_linear_fit_binary(binary_data_simple):
    """Test fitting binary SVMLinear."""
    X, y = binary_data_simple
    svm = SVMLinear(C=1.0, n_iters=100, random_state=42)
    
    svm.fit(X, y)
    
    # Check that model is marked as fitted
    assert svm.is_fitted_ is True
    assert svm.w is not None
    assert svm.b is not None
    assert svm.classes_ is not None


def test_svm_linear_fit_multiclass(multiclass_data):
    """Test fitting multiclass SVMLinear (One-vs-Rest)."""
    X, y = multiclass_data
    svm = SVMLinear(C=1.0, n_iters=100, random_state=42)
    
    svm.fit(X, y)
    
    # Check multiclass setup
    assert svm.is_fitted_ is True
    assert len(svm.classes_) == 3
    assert isinstance(svm.w, list)
    assert len(svm.w) == 3
    assert isinstance(svm.b, list)
    assert len(svm.b) == 3


def test_svm_linear_predict_binary(binary_data_simple):
    """Test binary prediction."""
    X, y = binary_data_simple
    svm = SVMLinear(C=1.0, n_iters=100, random_state=42)
    svm.fit(X, y)
    
    predictions = svm.predict(X)
    assert len(predictions) == len(X)
    assert all(p in [-1, 1] for p in predictions)


def test_svm_linear_predict_multiclass(multiclass_data):
    """Test multiclass prediction."""
    X, y = multiclass_data
    svm = SVMLinear(C=1.0, n_iters=100, random_state=42)
    svm.fit(X, y)
    
    predictions = svm.predict(X)
    assert len(predictions) == len(X)
    assert all(p in [0, 1, 2] for p in predictions)


def test_svm_linear_predict_with_string_labels(multiclass_data_labels):
    """Test prediction with non-numeric labels."""
    X, y = multiclass_data_labels
    svm = SVMLinear(C=1.0, n_iters=100, random_state=42)
    svm.fit(X, y)
    
    predictions = svm.predict(X)
    assert len(predictions) == len(X)
    assert all(p in ['cat', 'dog', 'bird'] for p in predictions)


def test_svm_linear_score_binary(binary_data_simple):
    """Test binary accuracy score."""
    X, y = binary_data_simple
    svm = SVMLinear(C=1.0, n_iters=100, random_state=42)
    svm.fit(X, y)
    
    score = svm.score(X, y)
    assert 0.0 <= score <= 1.0
    # For training data, should have reasonable accuracy
    assert score > 0.5


def test_svm_linear_score_multiclass(multiclass_data):
    """Test multiclass accuracy score."""
    X, y = multiclass_data
    svm = SVMLinear(C=1.0, n_iters=100, random_state=42)
    svm.fit(X, y)
    
    score = svm.score(X, y)
    assert 0.0 <= score <= 1.0


def test_svm_linear_input_validation_empty():
    """Test that empty inputs raise ValueError."""
    svm = SVMLinear(C=1.0)
    
    with pytest.raises(ValueError, match="cannot be empty"):
        svm.fit([], [])


def test_svm_linear_input_validation_mismatch():
    """Test that mismatched X and y shapes raise ValueError."""
    svm = SVMLinear(C=1.0)
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1])  # Wrong size
    
    with pytest.raises(ValueError, match="same number of samples"):
        svm.fit(X, y)


def test_svm_linear_predict_before_fit():
    """Test that predicting before fitting raises error."""
    svm = SVMLinear(C=1.0)
    with pytest.raises(AttributeError):
        svm.predict([[1, 2]])


def test_svm_linear_predict_empty():
    """Test that predicting on empty data raises ValueError."""
    X, y = np.array([[1, 2]]), np.array([1])
    svm = SVMLinear(C=1.0)
    svm.fit(X, y)
    
    with pytest.raises(ValueError, match="cannot be empty"):
        svm.predict([])


def test_svm_linear_method_chaining(binary_data_simple):
    """Test method chaining: fit returns self."""
    X, y = binary_data_simple
    svm = SVMLinear(C=1.0)
    result = svm.fit(X, y)
    
    assert result is svm


def test_svm_linear_repr():
    """Test __repr__ method."""
    svm = SVMLinear(C=2.0, n_iters=500, random_state=42)
    repr_str = repr(svm)
    
    assert "SVMLinear" in repr_str
    assert "C=2.0" in repr_str


# ─────────────────────────────────────────────────────────────────────
# Tests: RBF Kernel
# ─────────────────────────────────────────────────────────────────────

def test_rbf_kernel_shape():
    """Test RBF kernel output shape."""
    X = np.array([[0, 0], [1, 1]], dtype=float)
    Y = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    
    K = rbf_kernel(X, Y, gamma=1.0)
    
    assert K.shape == (2, 3)
    assert isinstance(K, np.ndarray)


def test_rbf_kernel_symmetry():
    """Test that RBF kernel is symmetric when X == Y."""
    X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    
    K = rbf_kernel(X, X, gamma=1.0)
    
    assert np.allclose(K, K.T)


def test_rbf_kernel_diagonal_is_one():
    """Test that diagonal of K(X, X) is all ones."""
    X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    
    K = rbf_kernel(X, X, gamma=1.0)
    
    assert np.allclose(np.diag(K), 1.0)


def test_rbf_kernel_decreases_with_distance():
    """Test that kernel value decreases with distance."""
    X = np.array([[0, 0]], dtype=float)
    Y = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    
    K = rbf_kernel(X, Y, gamma=1.0)
    
    # Should be monotonically decreasing
    assert K[0, 0] > K[0, 1] > K[0, 2]


def test_rbf_kernel_gamma_effect():
    """Test that larger gamma makes kernel decay faster."""
    X = np.array([[0, 0]], dtype=float)
    Y = np.array([[0, 0], [1, 0]], dtype=float)
    
    K_gamma_1 = rbf_kernel(X, Y, gamma=1.0)
    K_gamma_10 = rbf_kernel(X, Y, gamma=10.0)
    
    # For non-zero distance, larger gamma should give smaller value
    assert K_gamma_1[0, 1] > K_gamma_10[0, 1]


# ─────────────────────────────────────────────────────────────────────
# Tests: SMO Algorithm
# ─────────────────────────────────────────────────────────────────────

def test_smo_output_shape(binary_data_simple):
    """Test SMO returns correct shapes."""
    X, y = binary_data_simple
    K = rbf_kernel(X, X, gamma=1.0)
    
    alphas, b = smo(K, y, C=1.0, max_iter=10)
    
    assert alphas.shape == (len(X),)
    assert isinstance(b, (float, np.floating))


def test_smo_alphas_in_bounds(binary_data_simple):
    """Test that SMO output satisfies bounds."""
    X, y = binary_data_simple
    K = rbf_kernel(X, X, gamma=1.0)
    
    alphas, b = smo(K, y, C=1.0, max_iter=50)
    
    # Alphas should be in [0, C]
    assert np.all(alphas >= 0.0)
    assert np.all(alphas <= 1.0)


def test_smo_convergence(binary_data_simple):
    """Test SMO convergence on simple data."""
    X, y = binary_data_simple
    K = rbf_kernel(X, X, gamma=1.0)
    
    alphas, b = smo(K, y, C=1.0, tol=1e-3, max_iter=100)
    
    # For well-separated data, should converge
    assert np.any(alphas > 0)  # Some support vectors


# ─────────────────────────────────────────────────────────────────────
# Tests: SVMRBF (Binary RBF SVM)
# ─────────────────────────────────────────────────────────────────────

def test_svm_rbf_initialization():
    """Test SVMRBF initialization."""
    svm = SVMRBF(C=2.0, gamma=0.5)
    assert svm.C == 2.0
    assert svm.gamma == 0.5
    assert svm.is_fitted_ is False


def test_svm_rbf_fit(binary_data_simple):
    """Test SVMRBF fit."""
    X, y = binary_data_simple
    svm = SVMRBF(C=1.0, gamma=1.0)
    
    svm.fit(X, y)
    
    assert svm.is_fitted_ is True
    assert svm.support_vectors is not None
    assert svm.support_alphas is not None
    assert svm.support_labels is not None
    assert svm.b is not None


def test_svm_rbf_predict(binary_data_simple):
    """Test SVMRBF predict."""
    X, y = binary_data_simple
    svm = SVMRBF(C=1.0, gamma=1.0)
    svm.fit(X, y)
    
    predictions = svm.predict(X)
    
    assert len(predictions) == len(X)
    assert np.all(np.isin(predictions, [-1, 1]))


def test_svm_rbf_predict_before_fit():
    """Test that predicting before fitting raises error."""
    svm = SVMRBF(C=1.0, gamma=1.0)
    with pytest.raises(AttributeError):
        svm.predict([[1, 2]])


def test_svm_rbf_score(binary_data_simple):
    """Test SVMRBF score."""
    X, y = binary_data_simple
    svm = SVMRBF(C=1.0, gamma=1.0)
    svm.fit(X, y)
    
    score = svm.score(X, y)
    
    assert 0.0 <= score <= 1.0


def test_svm_rbf_gamma_effect(binary_data_simple):
    """Test effect of gamma parameter."""
    X, y = binary_data_simple
    
    svm_low_gamma = SVMRBF(C=1.0, gamma=0.1)
    svm_low_gamma.fit(X, y)
    
    svm_high_gamma = SVMRBF(C=1.0, gamma=10.0)
    svm_high_gamma.fit(X, y)
    
    # Both should fit successfully
    assert svm_low_gamma.is_fitted_
    assert svm_high_gamma.is_fitted_


# ─────────────────────────────────────────────────────────────────────
# Tests: SVMRBFOvO (Multiclass RBF SVM)
# ─────────────────────────────────────────────────────────────────────

def test_svm_rbf_ovo_initialization():
    """Test SVMRBFOvO initialization."""
    svm = SVMRBFOvO(C=2.0, gamma=0.5)
    assert svm.C == 2.0
    assert svm.gamma == 0.5
    assert svm.is_fitted_ is False


def test_svm_rbf_ovo_fit(multiclass_data):
    """Test SVMRBFOvO fit."""
    X, y = multiclass_data
    svm = SVMRBFOvO(C=1.0, gamma=1.0)
    
    svm.fit(X, y)
    
    assert svm.is_fitted_ is True
    # One-vs-One creates C(n_classes, 2) classifiers
    # For 3 classes: C(3,2) = 3 classifiers
    assert len(svm.classifiers) == 3


def test_svm_rbf_ovo_predict(multiclass_data):
    """Test SVMRBFOvO predict."""
    X, y = multiclass_data
    svm = SVMRBFOvO(C=1.0, gamma=1.0)
    svm.fit(X, y)
    
    predictions = svm.predict(X)
    
    assert len(predictions) == len(X)
    assert np.all(np.isin(predictions, [0, 1, 2]))


def test_svm_rbf_ovo_predict_one(multiclass_data):
    """Test SVMRBFOvO predict_one for single sample."""
    X, y = multiclass_data
    svm = SVMRBFOvO(C=1.0, gamma=1.0)
    svm.fit(X, y)
    
    prediction = svm.predict_one(X[0])
    
    assert prediction in [0, 1, 2]


def test_svm_rbf_ovo_predict_before_fit():
    """Test that predicting before fitting raises error."""
    svm = SVMRBFOvO(C=1.0, gamma=1.0)
    with pytest.raises(AttributeError):
        svm.predict([[1, 2]])


def test_svm_rbf_ovo_score(multiclass_data):
    """Test SVMRBFOvO score."""
    X, y = multiclass_data
    svm = SVMRBFOvO(C=1.0, gamma=1.0)
    svm.fit(X, y)
    
    score = svm.score(X, y)
    
    assert 0.0 <= score <= 1.0


def test_svm_rbf_ovo_majority_voting(multiclass_data):
    """Test that OvO uses majority voting correctly."""
    X, y = multiclass_data
    svm = SVMRBFOvO(C=1.0, gamma=1.0)
    svm.fit(X, y)
    
    # Single prediction should work and give a valid class
    pred = svm.predict_one(X[0])
    assert pred in np.unique(y)


# ─────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────

def test_svm_linear_vs_rbf_same_data(binary_data_simple):
    """Compare Linear and RBF SVMs on same data."""
    X, y = binary_data_simple
    
    svm_linear = SVMLinear(C=1.0, random_state=42)
    svm_linear.fit(X, y)
    
    svm_rbf = SVMRBF(C=1.0, gamma=1.0)
    svm_rbf.fit(X, y)
    
    # Both should produce predictions
    pred_linear = svm_linear.predict(X)
    pred_rbf = svm_rbf.predict(X)
    
    assert len(pred_linear) == len(pred_rbf) == len(X)


def test_different_c_values(binary_data_simple):
    """Test effect of C parameter."""
    X, y = binary_data_simple
    
    svm_small_c = SVMLinear(C=0.1, n_iters=100, random_state=42)
    svm_small_c.fit(X, y)
    
    svm_large_c = SVMLinear(C=10.0, n_iters=100, random_state=42)
    svm_large_c.fit(X, y)
    
    # Both should fit successfully
    assert svm_small_c.is_fitted_
    assert svm_large_c.is_fitted_


def test_reproducibility_with_random_state(binary_data_simple):
    """Test that same random_state produces same results."""
    X, y = binary_data_simple
    
    svm1 = SVMLinear(C=1.0, random_state=42)
    svm1.fit(X, y)
    pred1 = svm1.predict(X)
    
    svm2 = SVMLinear(C=1.0, random_state=42)
    svm2.fit(X, y)
    pred2 = svm2.predict(X)
    
    assert np.array_equal(pred1, pred2)


def test_list_input_conversion(binary_data_simple):
    """Test that both list and numpy array inputs work."""
    X_arr, y_arr = binary_data_simple
    X_list = X_arr.tolist()
    y_list = y_arr.tolist()
    
    svm = SVMLinear(C=1.0, random_state=42)
    svm.fit(X_list, y_list)
    
    predictions = svm.predict(X_list)
    assert len(predictions) == len(X_list)

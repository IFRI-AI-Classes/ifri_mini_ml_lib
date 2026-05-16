import numpy as np
import pytest 
from ifri_mini_ml_lib.classification.svm import SVM


#-----------------------------------------------------------------------------------------------------------------
# Données de test communes
#
# X_linear / y_linear : 8 points 2D parfaitement séparables par un hyperplan.
#
# X_rbf / y_rbf : 8 points 2D regroupés en deux ilots tres éloignés.
#------------------------------------------------------------------------------------------------------------------

X_linear = np.array([
    [1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [4.0, 5.0],   # classe  1
    [-1.0, -2.0], [-2.0, -3.0], [-3.0, -3.0], [-4.0, -5.0],  # classe -1
])
y_linear = np.array([1, 1, 1, 1, -1, -1, -1, -1])

X_rbf = np.array([
    [0.5, 0.5], [0.6, 0.4], [0.4, 0.6], [0.5, 0.6],   # classe  1
    [5.0, 5.0], [5.1, 4.9], [4.9, 5.1], [5.0, 4.8],   # classe -1
])
y_rbf = np.array([1, 1, 1, 1, -1, -1, -1, -1])

#============================================================================================================================================
# Tests d'initialisation
#============================================================================================================================================


def test_init_default():
    model = SVM()
    assert model.kernel == 'linear'
    assert model.C == 1.0
    assert model.gamma == 1.0
    assert not model.is_fitted_
    
def test_init_custom():
    model = SVM(kernel='rbf', C=0.5, gamma=2.0, max_iter=500, random_state=7)
    assert model.kernel == 'rbf'
    assert model.C == 0.5
    assert model.gamma == 2.0
    
def test_init_invalid_kernel():
    with pytest.raises(ValueError):
        SVM(kernel='sigmoid') 
        

#=============================================================================================================
# Tests kernel lineaire
#=============================================================================================================


def test_linear_fit_sets_is_fitted():
    model = SVM(kernel='linear', random_state=0)
    model.fit(X_linear, y_linear)
    assert model.is_fitted_
    
def test_linear_fit_returns_self():
    model = SVM(kernel='linear', random_state=0)
    result = model.fit(X_linear, y_linear)
    assert result is model
       
def test_linear_classes_detected():
    model = SVM(kernel='linear', random_state=0)
    model.fit(X_linear, y_linear)
    assert set(model.classes_) == {-1, 1}
        
def test_linear_weights_set():
    model = SVM(kernel='linear', random_state=0)
    model.fit(X_linear, y_linear)
    assert model.w_ is not None
    assert model.b_ is not None
        
def test_linear_predict_shape():
    model = SVM(kernel='linear', random_state=0)
    model.fit(X_linear, y_linear)
    preds = model.predict(X_linear)
    for p in preds:
        assert preds.shape == (len(y_linear),) 
               
def test_linear_accuracy():
    model = SVM(kernel='linear', C=1.0, max_iter=2000, random_state=0)
    model.fit(X_linear, y_linear)
    acc = np.mean(model.predict(X_linear) == y_linear)  
    assert acc >= 0.8

def test_linear_score():
    model = SVM(kernel='linear', C=1.0, max_iter=2000, random_state=0)
    model.fit(X_linear, y_linear)
    s = model.score(X_linear, y_linear)  
    assert 0.0 <= s <= 1.0
        
def test_linear_predict_before_fit_raises():
    model = SVM(kernel='linear')
    with pytest.raises(AttributeError):
        model.predict(X_linear)
               
def test_linear_fit_size_mismatch_raises():
    model = SVM(kernel='linear')
    with pytest.raises(ValueError):
        model.fit(X_linear, np.array([1, -1]))
        
        
# =============================================================================
# Tests kernel RBF (Radial Basis Function)
# =============================================================================


def test_rbf_fit_sets_is_fitted():
    # Après fit() avec kernel RBF, is_fitted_ doit passer à True
    model = SVM(kernel='rbf', C=1.0, gamma=1.0)
    model.fit(X_rbf, y_rbf)
    assert model.is_fitted_

def test_rbf_fit_returns_self():
    model = SVM(kernel='rbf', C=1.0, gamma=1.0)
    result = model.fit(X_rbf, y_rbf)
    assert result is model

def test_rbf_support_vectors_set():
    model = SVM(kernel='rbf', C=1.0, gamma=1.0)
    model.fit(X_rbf, y_rbf)
    assert model.support_vectors_ is not None
    assert model.support_alphas_ is not None
    assert model.support_labels_ is not None

def test_rbf_predict_shape():
    model = SVM(kernel='rbf', C=1.0, gamma=1.0)
    model.fit(X_rbf, y_rbf)
    preds = model.predict(X_rbf)
    assert preds.shape == (len(y_rbf),)

def test_rbf_predict_labels_valid():
    model = SVM(kernel='rbf', C=1.0, gamma=1.0)
    model.fit(X_rbf, y_rbf)
    preds = model.predict(X_rbf)
    for p in preds:
        assert p in [-1, 1]

def test_rbf_accuracy():
    # Sur deux groupes bien séparés, le kernel RBF doit atteindre
    # au moins 80% de précision sur les données d'entraînement
    model = SVM(kernel='rbf', C=2.0, gamma=1.0)
    model.fit(X_rbf, y_rbf)
    acc = np.mean(model.predict(X_rbf) == y_rbf)
    assert acc >= 0.8

def test_rbf_score():
    # score() doit renvoyer un float entre 0.0 et 1.0
    model = SVM(kernel='rbf', C=1.0, gamma=1.0)
    model.fit(X_rbf, y_rbf)
    s = model.score(X_rbf, y_rbf)
    assert 0.0 <= s <= 1.0

def test_rbf_predict_before_fit_raises():
    model = SVM(kernel='rbf')
    with pytest.raises(AttributeError):
        model.predict(X_rbf)
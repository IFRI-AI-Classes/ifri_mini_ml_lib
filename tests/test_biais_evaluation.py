import pytest
import numpy as np
from ifri_mini_ml_lib.classification.knn import KNN
from ifri_mini_ml_lib.metrics.bias_evaluation import equalized_odds_difference, equalized_odds_ratio, demographic_parity_difference, demographic_parity_ratio
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, selection_rate
from ifri_mini_ml_lib.preprocessing.preparation.min_max_scaler import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

@pytest.fixture
def breast_data():
    # Load the dataset as a DataFrame
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target.to_numpy()
    feature_names = data.feature_names
    return X, y, feature_names

@pytest.fixture
def knn_breast_data(breast_data):
    X, y, feature_names = breast_data

    # Standardize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.to_numpy()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Train the KNN model
    knn_model = KNN(k=5, task='classification')
    knn_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn_model.predict(X_test)

    # Create a sensitive attribute based on the "mean area" feature
    idx = list(feature_names).index('mean area')
    threshold = np.median(X_scaled[:, idx])
    sensitive_features = (X_test[:, idx] > threshold).astype(int)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    sensitive_features = np.array(sensitive_features)

    return y_test, y_pred, sensitive_features

def test_equalized_odds_difference_comparison(knn_breast_data):
    y_true, y_pred, sensitive_features = knn_breast_data

    # Compute with our custom equalized_odds_difference function
    diff_res, tpr_diff_dict, fpr_diff_dict = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features, pos_label=1
    )

    print("\n=== Results using our function ===")
    print("Equalized Odds Difference:", diff_res)
    print("TPR per group:", tpr_diff_dict)
    print("FPR per group:", fpr_diff_dict)

    # Use Fairlearn
    metrics = {
        'TPR': true_positive_rate,
        'FPR': false_positive_rate
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    # Compute differences with Fairlearn
    fairlearn_tpr = mf.by_group["TPR"]
    fairlearn_fpr = mf.by_group["FPR"]

    fairlearn_tpr_diff = max(fairlearn_tpr) - min(fairlearn_tpr)
    fairlearn_fpr_diff = max(fairlearn_fpr) - min(fairlearn_fpr)
    fairlearn_eo_diff = max(fairlearn_tpr_diff, fairlearn_fpr_diff)

    print("\n=== Results using Fairlearn ===")
    print("Fairlearn TPR Difference:", fairlearn_tpr_diff)
    print("Fairlearn FPR Difference:", fairlearn_fpr_diff)
    print("Fairlearn Equalized Odds Difference:", fairlearn_eo_diff)

    # Comparison
    assert np.isclose(diff_res, fairlearn_eo_diff, atol=1e-5), f"EO Difference mismatch: {diff_res} vs {fairlearn_eo_diff}"

    print("\n The results are similar between our function and Fairlearn.")

def test_equalized_odds_ratio_comparison(knn_breast_data):
    y_true, y_pred, sensitive_features = knn_breast_data

    # Compute with our custom equalized_odds_ratio function
    ratio_res, tpr_ratio_dict, fpr_ratio_dict = equalized_odds_ratio(
        y_true, y_pred, sensitive_features=sensitive_features, pos_label=1
    )

    print("\n=== Results using our function ===")
    print("Equalized Odds Ratio:", ratio_res)
    print("TPR per group:", tpr_ratio_dict)
    print("FPR per group:", fpr_ratio_dict)

    # Use Fairlearn
    metrics = {
        'TPR': true_positive_rate,
        'FPR': false_positive_rate
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    # Compute ratios with Fairlearn
    fairlearn_tpr = mf.by_group["TPR"]
    fairlearn_fpr = mf.by_group["FPR"]

    tpr_ratio = min(fairlearn_tpr) / max(fairlearn_tpr) if max(fairlearn_tpr) != 0 else 0
    fpr_ratio = min(fairlearn_fpr) / max(fairlearn_fpr) if max(fairlearn_fpr) != 0 else 0

    fairlearn_eo_ratio = min(tpr_ratio, fpr_ratio)

    print("\n=== Results using Fairlearn ===")
    print("Fairlearn TPR Ratio:", tpr_ratio)
    print("Fairlearn FPR Ratio:", fpr_ratio)
    print("Fairlearn Equalized Odds Ratio:", fairlearn_eo_ratio)

    # Comparison
    assert np.isclose(ratio_res, fairlearn_eo_ratio, atol=1e-5), f"EO Ratio mismatch: {ratio_res} vs {fairlearn_eo_ratio}"
    print("\n The results are similar between our function and Fairlearn.")
    

def test_demographic_parity_difference_comparison(knn_breast_data):
    y_true, y_pred, sensitive_features = knn_breast_data

    # Compute with our custom demographic_parity_difference function
    dp_diff, rate_dict_diff = demographic_parity_difference(y_pred, sensitive_features=sensitive_features, pos_label=1)
    
    print("\n=== Results using our function ===")
    print("Demographic Parity Difference:", dp_diff)
    print("Selection Rate per group:", rate_dict_diff)

    # Use Fairlearn
    mf = MetricFrame(
        metrics=selection_rate,
        y_true= y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    fairlearn_diff = mf.difference(method='between_groups')

    print("\n=== Results using Fairlearn ===")
    print("Fairlearn Demographic Parity Difference:", fairlearn_diff)

    # Comparison 
    assert np.isclose(dp_diff, fairlearn_diff, atol=1e-5), f"Difference mismatch: {dp_diff} vs {fairlearn_diff}"

    print("\n The results are similar between our function and Fairlearn.")

def test_demographic_parity_ratio_comparison(knn_breast_data):
    y_true, y_pred, sensitive_features = knn_breast_data

    # Compute with our custom demographic_parity_ratio function
    dp_ratio, rate_dict_ratio = demographic_parity_ratio(y_pred, sensitive_features)

    print("\n=== Results using our function ===")
    print("Demographic Parity Ratio:", dp_ratio)
    print("Selection Rate per group:", rate_dict_ratio)

    # Use Fairlearn 
    mf = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    fairlearn_ratio = mf.ratio(method='between_groups')

    print("\n=== Results using Fairlearn ===")
    print("Fairlearn Demographic Parity Ratio:", fairlearn_ratio)

    # Comparison 
    assert np.isclose(dp_ratio, fairlearn_ratio, atol=1e-5), f"Ratio mismatch: {dp_ratio} vs {fairlearn_ratio}"

    print("\n The results are similar between our function and Fairlearn.")

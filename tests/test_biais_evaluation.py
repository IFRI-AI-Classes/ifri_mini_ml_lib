import pytest
import numpy as np
import pandas as pd
from ifri_mini_ml_lib.classification.knn import KNN
from ifri_mini_ml_lib.metrics.bias_evaluation import equalized_odds_difference, equalized_odds_ratio, demographic_parity_difference, demographic_parity_ratio
from ifri_mini_ml_lib.preprocessing.preparation.min_max_scaler import MinMaxScaler
from ifri_mini_ml_lib.preprocessing.preparation.data_splitter import DataSplitter

@pytest.fixture
def breast_data():
    # Load the dataset as a DataFrame
    data = pd.read_csv("tests/breast-cancer.csv", sep=",")
    data.drop(columns="id", inplace=True)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    X = data.drop(columns="diagnosis")
    y = data["diagnosis"]
    feature_names = data.columns
    return X, y, feature_names

@pytest.fixture
def knn_breast_data(breast_data):
    X, y, feature_names = breast_data

    # Standardize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    splitter = DataSplitter(seed=42)
    X_train, X_test, y_train, y_test = splitter.train_test_split(X_scaled, y, test_size=0.3)
    X_train = X_train.values.astype(float)
    X_test = X_test.values.astype(float)

    # Train the KNN model
    knn_model = KNN(k=5, task='classification')
    knn_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn_model.predict(X_test)

    # Create a sensitive attribute based on the "mean area" feature
    idx = list(feature_names).index('area_mean')
    threshold = np.median(X_scaled.iloc[:, idx])
    sensitive_features = (X_test[:, idx] > threshold).astype(int)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    sensitive_features = np.array(sensitive_features)

    return y_test, y_pred, sensitive_features

def test_equalized_odds_difference_comparison(knn_breast_data):
    y_true, y_pred, sensitive_features = knn_breast_data


    diff_res, tpr_diff_dict, fpr_diff_dict = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features, pos_label=1
    )

    print("\n=== Results using our function ===")
    print("Equalized Odds Difference:", diff_res)
    print("TPR per group:", tpr_diff_dict)
    print("FPR per group:", fpr_diff_dict)

    assert 0 <= diff_res <= 1, "The result should be between 0 and 1"

def test_equalized_odds_ratio_comparison(knn_breast_data):
    y_true, y_pred, sensitive_features = knn_breast_data

    ratio_res, tpr_ratio_dict, fpr_ratio_dict = equalized_odds_ratio(
        y_true, y_pred, sensitive_features=sensitive_features, pos_label=1
    )

    print("\n=== Results using our function ===")
    print("Equalized Odds Ratio:", ratio_res)
    print("TPR per group:", tpr_ratio_dict)
    print("FPR per group:", fpr_ratio_dict)

    assert 0 <= ratio_res <= 1, "The result should be between 0 and 1"
    

def test_demographic_parity_difference_comparison(knn_breast_data):
    _, y_pred, sensitive_features = knn_breast_data

    dp_diff, rate_dict_diff = demographic_parity_difference(y_pred, sensitive_features=sensitive_features, pos_label=1)
    
    print("\n=== Results using our function ===")
    print("Demographic Parity Difference:", dp_diff)
    print("Selection Rate per group:", rate_dict_diff)

    assert 0 <= dp_diff <= 1, "The result should be between 0 and 1"

def test_demographic_parity_ratio_comparison(knn_breast_data):
    _, y_pred, sensitive_features = knn_breast_data

    dp_ratio, rate_dict_ratio = demographic_parity_ratio(y_pred, sensitive_features)

    print("\n=== Results using our function ===")
    print("Demographic Parity Ratio:", dp_ratio)
    print("Selection Rate per group:", rate_dict_ratio)

    assert 0 <= dp_ratio <= 1, "The result should be between 0 and 1"

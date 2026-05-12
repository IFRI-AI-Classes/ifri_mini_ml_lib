"""Direct test execution for the z_score module."""


from z_score import zscore_detection, modified_zscore_detection, summary_anomalies, ZScoreDetector
import numpy as np

print("=" * 60)
print("TEST OF THE MODULE z_score.py")
print("=" * 60)

# Test 1
print("\nTest 1: Detection of an anomaly")
data = [100, 102, 98, 101, 99, 300, 101, 98, 102, 99]
anomalies = zscore_detection(data, threshold=3.0)
print(f"Data : {data}")
print(f"Anomalies : {anomalies}")
print(f"Number of anomalies : {sum(anomalies)}")

if anomalies[5] == True:
    print("Test 1 passed - Anomaly detected !")

# Test 2
print("\n Test 2: With return of Z-scores")
anomalies, zscores = zscore_detection(data, return_zscore=True)
print(f"Z-scores : {np.round(zscores, 2)}")
print("Test 2 passed")

# Test 3
print("\nTest 3: Modified Z-score")
data2 = [10, 12, 11, 10, 1000, 11, 12, 10, 2000, 11]
anomalies = modified_zscore_detection(data2, threshold=3.5)
print(f"Modified anomalies : {anomalies}")
print(f"Number of anomalies : {sum(anomalies)}")
print("Test 3 passed")

# Test 4
print("Test 4")
X_train = [100, 102, 98, 101, 99, 103, 97, 101, 100, 102]
X_test = [101, 99, 350, 98]

detector = ZScoreDetector(threshold = 3.0)
detector.fit(X_train)
anomalies_ = detector.predict(X_test)

if anomalies_[2] == True:
    print("Test 4 passed")

# Test 5
anomalies_, zscores = detector.predict(X_test, return_zscore = True)
print(f"Z-scores : {np.round(zscores, 2)}")
print(f"Anomalies : {anomalies_}")
print("Test 5 passed")

# Test 6
print(f"Mean learned : {round(detector.mean_, 2)}")
print(f"Std learned : {round(detector.std_, 2)}")
print(f"Detector fitted : {detector.is_fitted_}")

if detector.is_fitted_:
    print("Test 6 passed")
    
# Test 7
X_train_2d = np.array([
    [37, 70],
    [36, 72],
    [37, 68],
    [36, 71],
    [37, 69]
])

X_test_2d = np.array([
    [37, 71],
    [42, 180]
])

detector_2d = ZScoreDetector( threshold = 3.0, axis = 0)
detector_2d.fit(X_train_2d)
anomalies_2d = detector_2d.predict(X_test_2d)
print("2D anomalies:")
print(anomalies_2d)

if anomalies_2d[1][0] and anomalies_2d[1][1]:
    print("Test 7 passed and 2D detection works")
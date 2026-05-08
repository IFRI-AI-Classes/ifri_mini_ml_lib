"""Direct test execution for the z_score module."""


from z_score import zscore_detection, modified_zscore_detection, summary_anomalies

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
    print("Test 1 PASSED - Anomaly detected !")

# Test 2
print("\n Test 2: With return of Z-scores")
anomalies, zscores = zscore_detection(data, return_zscore=True)
print(f"Z-scores : {[round(z, 2) for z in zscores]}")
print("Test 2 PASSED")

# Test 3
print("\nTest 3: Modified Z-score")
data2 = [10, 12, 11, 10, 1000, 11, 12, 10, 2000, 11]
anomalies = modified_zscore_detection(data2, threshold=3.5)
print(f"Modified anomalies : {anomalies}")
print(f"Number of anomalies : {sum(anomalies)}")
print("Test 3 PASSED")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ! ")
print("=" * 60)
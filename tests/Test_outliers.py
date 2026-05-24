import numpy as np
import pandas as pd
from ifri_mini_ml_lib.preprocessing.outlier_management.Treatment.univariate_treatment import Univariate_Treatment
from ifri_mini_ml_lib.preprocessing.outlier_management.Treatment.multivariate_treatment import Multivariate_Treatment


# Helpers 

def ok(msg):   print(f"  YES: {msg}")
def fail(msg): print(f"  NO:  {msg}")

def assert_true(cond, msg):
    if cond: ok(msg)
    else:    fail(msg)

def assert_equal(val, expected, msg):
    if val == expected: ok(msg)
    else: fail(f"{msg}  ->  got {val!r}, expected {expected!r}")


# test dataset 

def make_dataframe():
    # Row index 5 is the outlier (age=200, salary=-99999, score=999)
    # Rows 0-4 are clean inlier rows
    return pd.DataFrame({
        "age":    [22, 25, 27, 23, 24, 200],
        "salary": [3000, 3200, 2900, 3100, 3050, -99999],
        "score":  [88,   92,   85,   90,   87,   999],
        "name":   ["a",  "b",  "c",  "d",  "e",  "f"]  # non-numeric, must be ignored
    })

OUTLIER_INDICES = [5]



# UNIVARIATE TREATMENT TESTS


# Test 1.1 Constructor works on a copy of the original dataframe
df_original = make_dataframe()
treatment   = Univariate_Treatment(df_original)
treatment.dataframe.loc[0, "age"] = 9999

assert_equal(
    df_original.loc[0, "age"], 22,
    "modifying treatment.dataframe does not affect the original dataframe"
)
# Expected: YES  the original dataframe stays untouched because __init__ calls .copy()


# Test 1.2  remove_outliers drops the correct row
df     = make_dataframe()
tr     = Univariate_Treatment(df)
result = tr.remove_outliers(OUTLIER_INDICES)

assert_true(
    5 not in result.index,
    "remove_outliers: index 5 (outlier) is no longer in the result"
)
# Expected: YES  row 5 has been dropped

assert_equal(
    len(result), 5,
    "remove_outliers: exactly 5 rows remain (6 - 1 outlier)"
)
# Expected: YES  one row removed, five left

assert_equal(
    result.loc[0, "age"], 22,
    "remove_outliers: inlier rows are unchanged"
)
# Expected: YES  inlier values are not touched

assert_equal(
    len(df), 6,
    "remove_outliers: the original dataframe still has 6 rows"
)
# Expected: YES  the method works on a copy, original is intact


# Test 1.3 — winsorize caps values using inliers  bounds only
#  outlier_indices is now required bounds are computed on inliers only,
# so including the outlier row would bias the quantiles toward extreme values.
df     = make_dataframe()
tr     = Univariate_Treatment(df)
result = tr.winsorize(outlier_indices=OUTLIER_INDICES, lower_quantile=0.05, upper_quantile=0.95)

assert_equal(
    len(result), 6,
    "winsorize: no rows removed, still 6 rows"
)
# Expected: YES  winsorize replaces values, never removes rows

# Bounds must now be computed on inliers only (rows 0-4), not the full dataset
inlier_mask          = ~df.index.isin(OUTLIER_INDICES)
upper_bound_age      = df.loc[inlier_mask, "age"].quantile(0.95)
lower_bound_salary   = df.loc[inlier_mask, "salary"].quantile(0.05)

assert_true(
    result["age"].max() <= upper_bound_age + 1e-9,
    f"winsorize: max age ({result['age'].max():.1f}) <= inlier upper bound ({upper_bound_age:.1f})"
)
# Expected: YES  the outlier value 200 has been capped to the inlier 95th percentile

assert_true(
    result["salary"].min() >= lower_bound_salary - 1e-9,
    f"winsorize: min salary ({result['salary'].min():.1f}) >= inlier lower bound ({lower_bound_salary:.1f})"
)
# Expected: YES  the outlier value -99999 has been capped to the inlier 5th percentile

assert_equal(
    list(result["name"]), ["a", "b", "c", "d", "e", "f"],
    "winsorize: non-numeric column 'name' is unchanged"
)
# Expected: YES  _get_numeric_columns() excludes non-numeric columns


# Test 1.4 — impute_median replaces outliers with inlier median
df     = make_dataframe()
tr     = Univariate_Treatment(df)
result = tr.impute_median(OUTLIER_INDICES)

# Manually compute expected medians from inlier rows only (indices 0-4)
expected_median_age    = df.loc[:4, "age"].median()    # median of [22,25,27,23,24] = 24.0
expected_median_salary = df.loc[:4, "salary"].median() # median of [3000,3200,2900,3100,3050] = 3050.0

assert_equal(
    result.loc[5, "age"], expected_median_age,
    f"impute_median: age outlier replaced by inlier median ({expected_median_age})"
)
# Expected: YES  200 replaced by 24.0

assert_equal(
    result.loc[5, "salary"], expected_median_salary,
    f"impute_median: salary outlier replaced by inlier median ({expected_median_salary})"
)
# Expected: YES  -99999 replaced by 3050.0

assert_equal(
    result.loc[0, "age"], 22,
    "impute_median: inlier rows are unchanged"
)
# Expected: YES  only the outlier row is modified

assert_equal(
    len(result), 6,
    "impute_median: still 6 rows"
)
# Expected: YES  imputation never removes rows


# Test 1.5 — impute_mean replaces outliers with inlier mean
df     = make_dataframe()
tr     = Univariate_Treatment(df)
result = tr.impute_mean(OUTLIER_INDICES)

# Manually compute expected mean from inlier rows only (indices 0-4)
expected_mean_age = df.loc[:4, "age"].mean()  # (22+25+27+23+24)/5 = 24.2

assert_true(
    abs(result.loc[5, "age"] - expected_mean_age) < 1e-9,
    f"impute_mean: age outlier replaced by inlier mean ({expected_mean_age:.2f})"
)
# Expected: YES  200 replaced by 24.2

assert_equal(
    result.loc[0, "age"], 22,
    "impute_mean: inlier rows are unchanged"
)
# Expected: YES  only the outlier row is modified


# Test 1.6  log_transform applies log(1+x) to non-negative columns
df_pos = pd.DataFrame({
    "age":    [22, 25, 27, 23, 24, 200],
    "salary": [3000, 3200, 2900, 3100, 3050, 50000],
    "name":   ["a", "b", "c", "d", "e", "f"]
})
tr     = Univariate_Treatment(df_pos)
result = tr.log_transform()

expected_log = np.log1p(22)

assert_true(
    abs(result.loc[0, "age"] - expected_log) < 1e-9,
    f"log_transform: log(1+22) = {expected_log:.4f}"
)
# Expected: YES — log1p applied correctly

assert_true(
    result["age"].max() < df_pos["age"].max(),
    "log_transform: max value is reduced after transformation"
)
# Expected: YES — log compresses large values, 200 becomes ~5.3

assert_equal(
    list(result["name"]), ["a", "b", "c", "d", "e", "f"],
    "log_transform: non-numeric column 'name' is unchanged"
)
# Expected: YES  _get_numeric_columns() skips non-numeric columns

# Column with negative values must be skipped entirely
df_neg    = pd.DataFrame({"col": [-1, 2, 3, 200]})
tr_neg    = Univariate_Treatment(df_neg)
result_neg = tr_neg.log_transform()

assert_equal(
    list(result_neg["col"]), [-1, 2, 3, 200],
    "log_transform: column with negative values is skipped (left unchanged)"
)
# Expected: YES  log(negative) is undefined, the column is left as-is


# Test 1.7 — treat() dispatcher routes to the correct method
df = make_dataframe()
tr = Univariate_Treatment(df)

# All valid methods must be accepted without raising an exception
try:
    tr.treat("remove", outlier_indices=OUTLIER_INDICES)
    ok("dispatcher: 'remove' accepted")
except Exception as e:
    fail(f"dispatcher: 'remove' raised unexpected error -> {e}")
# Expected: YES

try:
    # winsorize now requires outlier_indices (bounds computed on inliers)
    tr.treat("winsorize", outlier_indices=OUTLIER_INDICES)
    ok("dispatcher: 'winsorize' accepted")
except Exception as e:
    fail(f"dispatcher: 'winsorize' raised unexpected error -> {e}")
# Expected: YES

try:
    tr.treat("impute_median", outlier_indices=OUTLIER_INDICES)
    ok("dispatcher: 'impute_median' accepted")
except Exception as e:
    fail(f"dispatcher: 'impute_median' raised unexpected error -> {e}")
# Expected: YES

# Unknown method must raise ValueError
try:
    tr.treat("unknown_method")
    fail("dispatcher: should have raised ValueError for unknown method")
except ValueError:
    ok("dispatcher: ValueError correctly raised for unknown method")
# Expected: YES — treat() raises ValueError for unrecognized method names



# MULTIVARIATE TREATMENT TESTS


# Test 2.1 — remove_outliers drops the correct row and keeps all columns
df     = make_dataframe()
tr     = Multivariate_Treatment(df)
result = tr.remove_outliers(OUTLIER_INDICES)

assert_true(5 not in result.index, "remove: index 5 dropped")
# Expected: YES  outlier row is removed

assert_equal(
    list(result.columns), list(df.columns),
    "remove: all columns are preserved including non-numeric"
)
# Expected: YES  removal never touches the column structure

assert_equal(len(result), 5, "remove: 5 rows remain")
# Expected: YES


# Test 2.2  impute_median replaces all numeric columns of the outlier row
df     = make_dataframe()
tr     = Multivariate_Treatment(df)
result = tr.impute_median(OUTLIER_INDICES)

expected_age    = df.loc[:4, "age"].median()    # 24.0
expected_salary = df.loc[:4, "salary"].median() # 3050.0
expected_score  = df.loc[:4, "score"].median()  # 88.0

assert_true(
    abs(result.loc[5, "age"] - expected_age) < 1e-9,
    f"impute_median multivariate: age -> {expected_age}"
)
# Expected: YES  all numeric columns of the outlier row are replaced simultaneously

assert_true(
    abs(result.loc[5, "salary"] - expected_salary) < 1e-9,
    f"impute_median multivariate: salary -> {expected_salary}"
)
# Expected: YES

assert_true(
    abs(result.loc[5, "score"] - expected_score) < 1e-9,
    f"impute_median multivariate: score -> {expected_score}"
)
# Expected: YES

assert_equal(result.loc[0, "age"], 22, "impute_median: inlier rows unchanged")
# Expected: YES — only the outlier row is modified


# Test 2.3  impute_knn replaces outlier with weighted average of k nearest inliers
df     = make_dataframe()
tr     = Multivariate_Treatment(df)
result = tr.impute_knn(OUTLIER_INDICES, n_neighbors=3)

assert_equal(len(result), 6, "impute_knn: still 6 rows")
# Expected: YES KNN imputation never removes rows

# Imputed value must fall within the inlier value range
inlier_age_min    = df.loc[:4, "age"].min()    # 22
inlier_age_max    = df.loc[:4, "age"].max()    # 27
inlier_salary_min = df.loc[:4, "salary"].min() # 2900
inlier_salary_max = df.loc[:4, "salary"].max() # 3200

assert_true(
    inlier_age_min <= result.loc[5, "age"] <= inlier_age_max,
    f"impute_knn: age imputed ({result.loc[5, 'age']:.2f}) within inlier range [{inlier_age_min}, {inlier_age_max}]"
)
# Expected: YES  weighted average of neighbors must stay within their range

assert_true(
    inlier_salary_min <= result.loc[5, "salary"] <= inlier_salary_max,
    f"impute_knn: salary imputed ({result.loc[5, 'salary']:.2f}) within inlier range [{inlier_salary_min}, {inlier_salary_max}]"
)
# Expected: YES

assert_equal(result.loc[0, "age"], 22, "impute_knn: inlier rows unchanged")
# Expected: YES

assert_equal(result.loc[1, "salary"], 3200, "impute_knn: inlier rows unchanged")
# Expected: YES

# With k=1, the imputed value must exactly equal the nearest neighbor's value
result_k1 = tr.impute_knn(OUTLIER_INDICES, n_neighbors=1)
assert_true(
    result_k1.loc[5, "age"] in df.loc[:4, "age"].values,
    "impute_knn k=1: imputed value equals the single nearest neighbor exactly"
)
# Expected: YES  with one neighbor, weight=1.0 so imputed = neighbor value exactly


# Test 2.4  treat() dispatcher routes to the correct method

#        because capping columns independently is not a true multivariate method.
#        True multivariate winsorizing requires Mahalanobis distance.
df = make_dataframe()
tr = Multivariate_Treatment(df)

for method, kwargs in [
    ("remove",        {"outlier_indices": OUTLIER_INDICES}),
    ("impute_median", {"outlier_indices": OUTLIER_INDICES}),
    ("impute_knn",    {"outlier_indices": OUTLIER_INDICES, "n_neighbors": 3}),
]:
    try:
        tr.treat(method, **kwargs)
        ok(f"dispatcher: '{method}' accepted")
    except Exception as e:
        fail(f"dispatcher: '{method}' raised error -> {e}")
# Expected: YES for all three methods

# winsorize must now raise ValueError (removed from multivariate)
try:
    tr.treat("winsorize", outlier_indices=OUTLIER_INDICES)
    fail("dispatcher: 'winsorize' should raise ValueError (removed from multivariate)")
except ValueError:
    ok("dispatcher: 'winsorize' correctly raises ValueError in multivariate context")
# Expected: YES — winsorize is no longer a valid multivariate method

# Unknown method must raise ValueError
try:
    tr.treat("unknown_method")
    fail("dispatcher: should have raised ValueError for unknown method")
except ValueError:
    ok("dispatcher: ValueError correctly raised for unknown method")
# Expected: YES


print("\nAll tests completed.")
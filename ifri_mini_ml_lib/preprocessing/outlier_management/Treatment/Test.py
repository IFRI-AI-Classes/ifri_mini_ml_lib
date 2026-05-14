import numpy as np
import pandas as pd
from univariate_treatment import Univariate_Treatment
from multivariate_treatment import Multivariate_Treatment

def ok(msg):   print(f"  ✅  {msg}")
def fail(msg): print(f"  ❌  {msg}")

def assert_true(cond, msg):
    if cond: ok(msg)
    else:    fail(msg)

def assert_equal(val, expected, msg):
    if val == expected: ok(msg)
    else: fail(f"{msg}  →  got {val!r}, expected {expected!r}")

def section(title):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print('═' * 60)
    
def make_dataframe():
    """
    Creates a simple test dataframe with 2 obvious outliers
    at index 5 (age=200, salary=-99999, score=999).
    Indices 0-4 are clean inlier rows.
    """
    return pd.DataFrame({
        "age":    [22, 25, 27, 23, 24, 200],
        "salary": [3000, 3200, 2900, 3100, 3050, -99999],
        "score":  [88,   92,   85,   90,   87,   999],
        "name":   ["a",  "b",  "c",  "d",  "e",  "f"]  # non-numeric ; must be ignored
    })

OUTLIER_INDICES = [5]


#UnivariateTreatment: the constructor is working on a copy

df_original = make_dataframe()
treatment   = Univariate_Treatment(df_original)

treatment.dataframe.loc[0, "age"] = 9999

assert_equal(
    df_original.loc[0, "age"], 22,
    "modifying treatment.dataframe does not affect the original dataframe"
)


#remove outliers
df   = make_dataframe()
tr   = Univariate_Treatment(df)
result = tr.remove_outliers(OUTLIER_INDICES)
assert_true(
    5 not in result.index,
    "remove_outliers: index 5 (outlier) is no longer in the result"
    
)

assert_equal(
    len(result), 5,
    "remove_outliers: there are exactly 5 lines left (6 - 1 outlier)"
)

print(result)

assert_equal(
    result.loc[0, "age"], 22,
    "remove_outliers: The inliers are unchanged."
    
)

assert_equal(
    len(df), 6,
    "remove_outliers:the original dataframe retains its 6 lines "
)


#WINSORIZE TEST
section("1.3 — UnivariateTreatment: winsorize")

df = make_dataframe()
tr = Univariate_Treatment(df)
result = tr.winsorize(lower_quantile=0.05, upper_quantile=0.95)

# 3a : le nombre de lignes est intact (winsorize ne supprime rien)
assert_equal(
    len(result), 6,
    "winsorize: aucune ligne supprimée, toujours 6 lignes"
)

# 3b : la valeur max de age ne dépasse plus le quantile 95%
upper_bound_age = df["age"].quantile(0.95)
assert_true(
    result["age"].max() <= upper_bound_age + 1e-9,
    f"winsorize: max age ({result['age'].max():.1f}) ≤ upper_bound ({upper_bound_age:.1f})"
)

# 3c : la valeur min de salary ne descend plus sous le quantile 5%
lower_bound_salary = df["salary"].quantile(0.05)
assert_true(
    result["salary"].min() >= lower_bound_salary - 1e-9,
    f"winsorize: min salary ({result['salary'].min():.1f}) ≥ lower_bound ({lower_bound_salary:.1f})"
)

# 3d : la colonne non-numérique 'name' est intacte
assert_equal(
    list(result["name"]), ["a", "b", "c", "d", "e", "f"],
    "winsorize: la colonne non-numérique 'name' est inchangée"
)

section("1.4 — UnivariateTreatment: impute_median")

df = make_dataframe()
tr = Univariate_Treatment(df)
result = tr.impute_median(OUTLIER_INDICES)

# Calcul manuel de la médiane attendue (inliers seulement = indices 0 à 4)
expected_median_age    = df.loc[:4, "age"].median()     # médiane de [22,25,27,23,24] = 24
expected_median_salary = df.loc[:4, "salary"].median()  # médiane de [3000,3200,2900,3100,3050] = 3050

# 4a : la valeur à l'index 5 est bien remplacée par la médiane
assert_equal(
    result.loc[5, "age"], expected_median_age,
    f"impute_median: age outlier remplacé par la médiane inlier ({expected_median_age})"
)

assert_equal(
    result.loc[5, "salary"], expected_median_salary,
    f"impute_median: salary outlier remplacé par la médiane inlier ({expected_median_salary})"
)

# 4b : les inliers ne sont pas touchés
assert_equal(
    result.loc[0, "age"], 22,
    "impute_median: les inliers sont inchangés"
)

# 4c : le nombre de lignes est intact
assert_equal(
    len(result), 6,
    "impute_median: toujours 6 lignes"
)


section("1.5 — UnivariateTreatment: impute_mean")

df = make_dataframe()
tr = Univariate_Treatment(df)
result = tr.impute_mean(OUTLIER_INDICES)

# Calcul manuel de la moyenne attendue (inliers seulement)
expected_mean_age = df.loc[:4, "age"].mean()  # (22+25+27+23+24)/5 = 24.2

# 5a : valeur remplacée par la moyenne inlier
assert_true(
    abs(result.loc[5, "age"] - expected_mean_age) < 1e-9,
    f"impute_mean: age outlier remplacé par la moyenne inlier ({expected_mean_age:.2f})"
)

# 5b : les inliers sont inchangés
assert_equal(
    result.loc[0, "age"], 22,
    "impute_mean: les inliers sont inchangés"
)

    
section("1.6 — UnivariateTreatment: log_transform")

# Dataset sans négatifs pour tester la transformation
df_pos = pd.DataFrame({
    "age":    [22, 25, 27, 23, 24, 200],
    "salary": [3000, 3200, 2900, 3100, 3050, 50000],
    "name":   ["a", "b", "c", "d", "e", "f"]
})
tr = Univariate_Treatment(df_pos)
result = tr.log_transform()

# 6a : vérifier log(1+x) sur une valeur connue
expected = np.log1p(22)
assert_true(
    abs(result.loc[0, "age"] - expected) < 1e-9,
    f"log_transform: log(1+22) = {expected:.4f} ✓"
)

# 6b : la valeur extrême est bien compressée
assert_true(
    result["age"].max() < df_pos["age"].max(),
    "log_transform: la valeur max est réduite après transformation"
)

# 6c : colonne non-numérique intacte
assert_equal(
    list(result["name"]), ["a", "b", "c", "d", "e", "f"],
    "log_transform: 'name' inchangé"
)

# 6d : colonne avec des négatifs → doit être skippée
df_neg = pd.DataFrame({"col": [-1, 2, 3, 200]})
tr_neg = Univariate_Treatment(df_neg)
result_neg = tr_neg.log_transform()

# La colonne doit rester inchangée car elle contient des négatifs
assert_equal(
    list(result_neg["col"]), [-1, 2, 3, 200],
    "log_transform: colonne avec négatifs ignorée (non transformée)"
)
    
section("1.7 — UnivariateTreatment: dispatcher treat()")

df = make_dataframe()
tr = Univariate_Treatment(df)

# Toutes les méthodes valides passent sans erreur
try:
    tr.treat("remove",       outlier_indices=OUTLIER_INDICES)
    ok("dispatcher: 'remove' accepté")
except Exception as e:
    fail(f"dispatcher: 'remove' a levé une erreur inattendue → {e}")

try:
    tr.treat("winsorize")
    ok("dispatcher: 'winsorize' accepté")
except Exception as e:
    fail(f"dispatcher: 'winsorize' a levé une erreur inattendue → {e}")

try:
    tr.treat("impute_median", outlier_indices=OUTLIER_INDICES)
    ok("dispatcher: 'impute_median' accepté")
except Exception as e:
    fail(f"dispatcher: 'impute_median' a levé une erreur inattendue → {e}")

# Méthode inconnue → doit lever ValueError
try:
    tr.treat("unknown_method")
    fail("dispatcher: aurait dû lever ValueError pour méthode inconnue")
except ValueError:
    ok("dispatcher: ValueError correctement levée pour méthode inconnue")
    
    
section("2.1 — MultivariateTreatment: remove_outliers")

df  = make_dataframe()
tr  = Multivariate_Treatment(df)
result = tr.remove_outliers(OUTLIER_INDICES)

# 8a : la ligne outlier est supprimée
assert_true(5 not in result.index, "remove: index 5 supprimé")

# 8b : toutes les colonnes sont préservées (y compris 'name')
assert_equal(
    list(result.columns), list(df.columns),
    "remove: toutes les colonnes sont préservées"
)

# 8c : les 5 inliers sont intacts
assert_equal(len(result), 5, "remove: 5 lignes restantes")

section("2.2 — MultivariateTreatment: impute_median")

df = make_dataframe()
tr = Multivariate_Treatment(df)
result = tr.impute_median(OUTLIER_INDICES)

# 9a : TOUTES les colonnes numériques de la ligne outlier sont remplacées
expected_age    = df.loc[:4, "age"].median()
expected_salary = df.loc[:4, "salary"].median()
expected_score  = df.loc[:4, "score"].median()

assert_true(
    abs(result.loc[5, "age"] - expected_age) < 1e-9,
    f"impute_median multivarié: age → {expected_age}"
)
assert_true(
    abs(result.loc[5, "salary"] - expected_salary) < 1e-9,
    f"impute_median multivarié: salary → {expected_salary}"
)
assert_true(
    abs(result.loc[5, "score"] - expected_score) < 1e-9,
    f"impute_median multivarié: score → {expected_score}"
)

# 9b : les inliers ne sont pas touchés
assert_equal(result.loc[0, "age"], 22, "impute_median: inliers inchangés")

section("2.3 — MultivariateTreatment: impute_knn")

df = make_dataframe()
tr = Multivariate_Treatment(df)
result = tr.impute_knn(OUTLIER_INDICES, n_neighbors=3)

# 10a : le nombre de lignes est intact
assert_equal(len(result), 6, "impute_knn: toujours 6 lignes")

# 10b : la valeur imputée est dans la plage des inliers
# (ne peut pas être plus grande que le max inlier ni plus petite que le min inlier)
inlier_age_min = df.loc[:4, "age"].min()  # 22
inlier_age_max = df.loc[:4, "age"].max()  # 27

assert_true(
    inlier_age_min <= result.loc[5, "age"] <= inlier_age_max,
    f"impute_knn: age imputé ({result.loc[5, 'age']:.2f}) "
    f"dans la plage inlier [{inlier_age_min}, {inlier_age_max}]"
)

# 10c : même vérification pour salary
inlier_salary_min = df.loc[:4, "salary"].min()  # 2900
inlier_salary_max = df.loc[:4, "salary"].max()  # 3200

assert_true(
    inlier_salary_min <= result.loc[5, "salary"] <= inlier_salary_max,
    f"impute_knn: salary imputé ({result.loc[5, 'salary']:.2f}) "
    f"dans la plage inlier [{inlier_salary_min}, {inlier_salary_max}]"
)

# 10d : les inliers ne sont pas touchés
assert_equal(result.loc[0, "age"], 22, "impute_knn: inliers inchangés")
assert_equal(result.loc[1, "salary"], 3200, "impute_knn: inliers inchangés")

# 10e : tester avec k=1 → valeur imputée = valeur du voisin le plus proche
result_k1 = tr.impute_knn(OUTLIER_INDICES, n_neighbors=1)
assert_true(
    result_k1.loc[5, "age"] in df.loc[:4, "age"].values,
    "impute_knn k=1: valeur imputée = valeur exacte du voisin le plus proche"
)



section("2.4 — MultivariateTreatment: dispatcher treat()")

df = make_dataframe()
tr = Multivariate_Treatment(df)

# Toutes les méthodes valides
for method, kwargs in [
    ("remove",        {"outlier_indices": OUTLIER_INDICES}),
    ("winsorize",     {}),
    ("impute_median", {"outlier_indices": OUTLIER_INDICES}),
    ("impute_knn",    {"outlier_indices": OUTLIER_INDICES, "n_neighbors": 3}),
]:
    try:
        tr.treat(method, **kwargs)
        ok(f"dispatcher: '{method}' accepté")
    except Exception as e:
        fail(f"dispatcher: '{method}' a levé une erreur → {e}")

# Méthode inconnue
try:
    tr.treat("mauvaise_methode")
    fail("dispatcher: aurait dû lever ValueError")
except ValueError:
    ok("dispatcher: ValueError levée pour méthode inconnue")


section("Tous les tests terminés")
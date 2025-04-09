import pandas as pd
import numpy as np
#from ..evaluation_metrics.Evaluation_metrique import confusion_matrix


def selection_rates(df, y_pred, privileged_group, unprivileged_group):
  """
  function (selection_rates) : measures the proportion of individuals in a given group who receive a positive decision

  privileged_group : group with the highest selection rate

  unprivilegied_group : group with the lowest selection rate and therefore disadvantaged

  args :  y_pred (list): predicted labels returned by the classifier
          df : dataframe containing predictions and groups

  return : the selection rates of the two groups
  """
  prob_privileged = df[df['group'] == privileged_group][y_pred].mean()
  prob_unprivileged = df[df['group'] == unprivileged_group][y_pred].mean()
  return prob_privileged, prob_unprivileged


def matrix (y_true, y_pred, group_mask):
    """
    function (matrix)  : establishes the confusion matrix for each group
    tn : true negative
    fp : false postive
    fn : false negative
    tp : true positive

    args : y_true (list) : truth labels provides to the classifier
            y_pred (list): predicted labels returned by the classifier
            group_mask (list) : Boolean mask indicating the elements belonging to
            the group for which we wish to calculate the confusion matrix

    return : tn, fp, fn, tp
    """
    # Appliquer le groupe mask pour obtenir les prédictions et les vraies valeurs pour le groupe
    y_true_group = [y_true[i] for i in range(len(y_true)) if group_mask[i]]
    y_pred_group = [y_pred[i] for i in range(len(y_pred)) if group_mask[i]]

    # Calcul de la matrice de confusion pour ce groupe
    conf_matrix_group = confusion_matrix(y_true_group, y_pred_group, classes=[0, 1])

    # Extraction des valeurs TN, FP, FN, TP
    tn = conf_matrix_group[0][0]  # Vrais négatifs
    fp = conf_matrix_group[0][1]  # Faux positifs
    fn = conf_matrix_group[1][0]  # Faux négatifs
    tp = conf_matrix_group[1][1]  # Vrais positifs

    return tn, fp, fn, tp


def group_rate (y_true, y_pred, group_mask):
  """
  function (group_rate): calculate the true positive and false positive rates for each group
  tpr : true positive rate
  fpr : false positive rate

    args : y_true (list) : truth labels provides to the classifier
            y_pred (list): predicted labels returned by the classifier
            group_mask (list) : Boolean mask indicating the elements belonging to
            the group for which we wish to calculate the confusion matrix

    return : tpr, fpr
    """
  tn, fp, fn, tp = matrix (y_true, y_pred, group_mask)
  tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
  fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

  return tpr, fpr


def ifri_demographic_parity_ratio(y_pred,privileged_group, unprivileged_group, group):
  """
  function : Calculate the demographic parity ratio between unprivileged and privileged groups
  args:

  privileged_group : group with the highest selection rate

  unprivilegied_group : group with the lowest selection rate and therefore disadvantaged

  group : list of groups

  return: demographic parity ratio (float)

  """
  # Conversion to data frame
  dict = {'y_pred': y_pred, 'group' : group}
  df = pd.DataFrame(dict)
  # Calculate the selection rate for each group
  prob_privileged, prob_unprivileged = selection_rates(df, y_pred)
  # Calculate the demographic parity ratio

  return prob_unprivileged / prob_privileged if prob_privileged != 0 else 0



def ifri_demographic_parity_difference(y_pred,priviligied_group, unpriviligied_group, group):
  """
  function : Calculate the demographic parity difference between the unpriviligied and priviligied groups
  args:
  y_pred (list): predicted labels returned by the classifier

  privileged_group : group with the highest selection rate

  unprivilegied_group : group with the lowest selection rate and therefore disadvantaged

  group : list of groups

  return: demographic parity difference (float)

  """
  # Conversion to data frame
  dict = {'y_pred': y_pred, 'group' : group}
  df = pd.DataFrame(dict)
  # Calculate the selection rate for each group
  prob_privileged, prob_unprivileged = selection_rates(df, y_pred)
  # Calculate the demographic parity difference

  return abs(prob_unprivileged - prob_privileged)


def ifri_equalized_odds(y_true, y_pred, privileged_group, unprivileged_group,sensitive_features):
  """
  function: Compare the rates of False Positives and True Positives between groups

  args:
   y_pred (list): predicted labels returned by the classifier

  privileged_group : group with the highest selection rate

  unprivilegied_group : group with the lowest selection rate and therefore disadvantaged

  sensitive_features : list of the sensitive features over which equal opportunity should be assessed

  return: the equalized odds ratio (float)

  """
  # Convert data to NumPy arrays
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  sensitive_features = np.array(sensitive_features)

  # Create the confusion matrices for each group
  group_0_mask = np.isin(sensitive_features, [unprivileged_group])
  group_1_mask = np.isin(sensitive_features, [privileged_group])

  # Calculate tpr and fpr of group_0
  tpr_0 , fpr_0 = group_rate(y_true, y_pred, group_0_mask )

  # Calculate tpr and fpr of group_1
  tpr_1 , fpr_1 = group_rate(y_true, y_pred, group_1_mask )

  tpr_ratio = tpr_1 / tpr_0 if tpr_0 > 0 else 0
  fpr_ratio = fpr_1 / fpr_0 if fpr_0 > 0 else 0

  return tpr_ratio, fpr_ratio



import pandas as pd
import numpy as np
#from ..evaluation_metrics.Evaluation_metrique import confusion_matrix


def selection_rates(df, y_pred, privileged_group, unprivileged_group):
  """
  Description: Selection rates is a that is function used to measure the proportion of individuals in a given group who receive a positive decision

  Args:  
      y_pred (list): predicted labels returned by the classifier
      df : dataframe containing predictions and groups
      privileged_group(float) : group with the highest selection rate
      unprivilegied_group(float) : group with the lowest selection rate and therefore disadvantaged

  Return: 
        the selection rates of the two groups
  
  Example:
       Assuming that you have a DataFrame 'df' with a 'group' column which indicates group membership, 
       a list of predicted labels 'y_pred' = [0,1,1,0,1,...] that you can call

       Then call on the function :
       result = selection_rates(df, y_pred, privileged_group, unprivileged_group); 
       print("Selection Rate:", result)
       it's going to return the proportion of positive predicitions for each group 

   """
  prob_privileged = df[df['group'] == privileged_group][y_pred].mean()
  prob_unprivileged = df[df['group'] == unprivileged_group][y_pred].mean()
  return prob_privileged, prob_unprivileged


def matrix (y_true, y_pred, group_mask):
    """
    Description: Matrix is a function thzt is used to establishe the confusion matrix for each group
    
    Args: 
        y_true (list) : truth labels provides to the classifier
        y_pred (list): predicted labels returned by the classifier
        group_mask (list) : Boolean mask indicating the elements belonging to the group for which we wish to calculate the confusion matrix

    Return:
        tn, fp, fn, tp
        tn : true negative
        fp : false postive
        fn : false negative
        tp : true positive

    Example: 
          Assuming that you have a list of truth labels 'y_true' and a list of predicted labels 'y_pred' that you'll applied group_mask on to obtain
          a list which will be used to calculate the confusion matrix

          And then, call on the function by doing :
          result = confusion_matrix(y_true_group, y_pred_group, classes=[0, 1])
          print("Confusion matrix:", result) 
          It's going to return tn, fp, fn, tp

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
  Description: Group Rate is a function that is used to calculate the true positive and false positive rates for each group
 
  Args:
       y_true (list) : truth labels provides to the classifier
       y_pred (list): predicted labels returned by the classifier 
       group_mask (list) : Boolean mask indicating the elements belonging to the group for which we wish to calculate the confusion matrix

  Return : tpr, fpr
      tpr : true positive rate
      fpr : false positive rate

  Example:
      Assuming that you have a list of truth labels 'y_true' and a list of predicted labels 'y_pred' that you'll applied group_mask on to obtain
      a list which will be used to calculate the confusion matrix

      And then, call on the function by doing :
          result = group_rate (y_true, y_pred, group_mask)
          print("Group Rate:", result) 
          It's going to return tpr, fpr for each group
  """
  tn, fp, fn, tp = matrix (y_true, y_pred, group_mask)
  tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
  fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

  return tpr, fpr


def demographic_parity_ratio(y_pred,privileged_group, unprivileged_group, group):
  """
  Description: This is a function used to calculate the demographic parity ratio between unprivileged and privileged groups
  This demographic parity is a metric of equity in AI that measures whether a model gives positibe predictions in an equitable 
  way between different groups

  Args:
      y_pred (list): predicted labels returned by the classifier 
      privileged_group(float): group with the highest selection rate
      unprivilegied_group(float): group with the lowest selection rate and therefore disadvantaged
      group(list): list of groups

  Return: 
      demographic parity ratio (float)

  Example:
      Assuming that you have a DataFrame 'df' with a 'group' column which indicates group membership, 
      a list of predicted labels 'y_pred' = [0,1,1,0,1,...] that you can call 

      And then, call on the function by doing :
      result = demographic_parity_ratio(y_pred,privileged_group, unprivileged_group, group)
      print(" demographic parity ratio:", result) 
      It's going to return the ratio of the prob_unprivileged by the prob_privileged

  """
  # Conversion to data frame
  dict = {'y_pred': y_pred, 'group' : group}
  df = pd.DataFrame(dict)
  # Calculate the selection rate for each group
  prob_privileged, prob_unprivileged = selection_rates(df, y_pred)
  
  return prob_unprivileged / prob_privileged if prob_privileged != 0 else 0



def demographic_parity_difference(y_pred,priviligied_group, unpriviligied_group, group):
  """
  Description: This is a function used to calculate the demographic parity difference between unprivileged and privileged groups
  This demographic parity difference is a metric of equity in AI that measures whether a model gives positibe predictions in an equitable 
  way between different groups. And this by making the absolute difference between this different groups.

  Args:
      y_pred (list): predicted labels returned by the classifier
      privileged_group : group with the highest selection rate
      unprivilegied_group : group with the lowest selection rate and therefore disadvantaged
      group : list of groups

  Return: 
      demographic parity difference (float)

  Example: 
      Assuming that you have a DataFrame 'df' with a 'group' column which indicates group membership, 
      a list of predicted labels 'y_pred' = [0,1,1,0,1,...] that you can call 

      And then, call on the function by doing :
      result = demographic_parity_difference(y_pred,privileged_group, unprivileged_group, group)
      print(" demographic parity diffence:", result) 
      It's going to return the absolute difference between the prob_unprivileged and the prob_privileged

  """
  # Conversion to data frame
  dict = {'y_pred': y_pred, 'group' : group}
  df = pd.DataFrame(dict)
  # Calculate the selection rate for each group
  prob_privileged, prob_unprivileged = selection_rates(df, y_pred)
  # Calculate the demographic parity difference

  return abs(prob_unprivileged - prob_privileged)


def equalized_odds(y_true, y_pred, privileged_group, unprivileged_group,sensitive_features):
  """
  Description: This is a function used to compare the rates of False Positives and True Positives between groups
  Equalized odd is an algorithmic equity metric that assesses whether a binary classification model treats different groups fairly

  Args:
      y_pred (list): predicted labels returned by the classifier
      privileged_group(float) : group with the highest selection rate
      unprivilegied_group(float) : group with the lowest selection rate and therefore disadvantage
      sensitive_features(list) : list of the sensitive features over which equal opportunity should be assessed

  Return: 
      the equalized odds ratio (float)

  Example: 
    Assuming that you have a list of truth labels 'y_true' and a list of predicted labels 'y_pred' that you'll applied group_mask on to obtain
    a list which will be used to calculate the confusion matrix for each group

    And then, call on the function group_rate (y_true, y_pred, group_mask) to obtain the tpr and the fpr of each group
    
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



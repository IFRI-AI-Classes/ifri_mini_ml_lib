def confusion_matrix(y_true, y_pred, classes=None):
    """
    Computes the confusion matrix.

    Args:
        y_true (list[int]): List of actual (true) labels.
        y_pred (list[int]): List of predicted labels.
        classes (list[int], optional): List of possible classes. 
                                       If None, classes are inferred from the data.

    Returns:
        dict: Confusion matrix as a nested dictionary in the form:
              {true_class: {predicted_class: count}}.
              Example: {0: {0: 50, 1: 10}, 1: {0: 5, 1: 100}}.
    """
    if classes is None:
        classes = sorted(set(y_true) | set(y_pred))  # Get all unique classes

    # Initialize the matrix with zeros
    matrix = {true_class: {pred_class: 0 for pred_class in classes}
              for true_class in classes}

    # Fill in the matrix with counts
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1

    return matrix


def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of predictions.

    Args:
        y_true (list[int]): List of true labels.
        y_pred (list[int]): List of predicted labels.

    Returns:
        float: Accuracy score (correct predictions / total predictions).
    """
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def precision(y_true, y_pred, positive_class=1):
    """
    Calculates precision for a specific class.

    Args:
        y_true (list[int]): List of true labels.
        y_pred (list[int]): List of predicted labels.
        positive_class (int, optional): The class considered as positive (default is 1).

    Returns:
        float: Precision score = True Positives / (True Positives + False Positives).
               Returns 0 if there are no predicted positives.
    """
    tp = fp = 0  # Initialize True Positives and False Positives

    for true, pred in zip(y_true, y_pred):
        if pred == positive_class:
            if true == positive_class:
                tp += 1  # True Positive
            else:
                fp += 1  # False Positive

    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall(y_true, y_pred, positive_class=1):
    """
    Calculates recall (sensitivity) for a specific class.

    Args:
        y_true (list[int]): List of true labels.
        y_pred (list[int]): List of predicted labels.
        positive_class (int, optional): The class considered as positive (default is 1).

    Returns:
        float: Recall score = True Positives / (True Positives + False Negatives).
               Returns 0 if there are no actual positives.
    """
    tp = fn = 0  # Initialize True Positives and False Negatives

    for true, pred in zip(y_true, y_pred):
        if true == positive_class:
            if pred == positive_class:
                tp += 1  # True Positive
            else:
                fn += 1  # False Negative

    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(y_true, y_pred, positive_class=1):
    """
    Calculates the F1-score for a specific class.

    Args:
        y_true (list[int]): List of true labels.
        y_pred (list[int]): List of predicted labels.
        positive_class (int, optional): The class considered as positive (default is 1).

    Returns:
        float: F1-score = 2 * (precision * recall) / (precision + recall).
               Returns 0 if both precision and recall are 0.
    """
    prec = precision(y_true, y_pred, positive_class)
    rec = recall(y_true, y_pred, positive_class)

    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

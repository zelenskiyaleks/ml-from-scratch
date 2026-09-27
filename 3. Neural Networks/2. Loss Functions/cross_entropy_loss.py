import numpy as np


def cross_entropy_loss(
    y_true: list[int],
    y_pred: list[list[float]]
) -> float:
    """
    Calculate the mean multiclass cross-entropy loss.

    Args:
        y_true: True class indices for each sample.
        y_pred: Predicted class probabilities.

    Returns:
        Mean cross-entropy loss as a Python float.

    Formula:
        L = -1/N * sum(log(p_true))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    samples = np.arange(len(y_true))
    true_class_probabilities = y_pred[samples, y_true]

    return float(-np.log(true_class_probabilities).mean())
import numpy as np


def mean_squared_error(
    y_pred: list,
    y_true: list
) -> float:
    """
    Calculate the mean squared error between predictions and targets.

    Args:
        y_pred: Predicted values.
        y_true: True target values.

    Returns:
        Mean squared error as a float.

    Formula:
        MSE = 1 / N * sum((y_pred - y_true)^2)
    """

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    return float(np.mean((y_pred - y_true) ** 2))
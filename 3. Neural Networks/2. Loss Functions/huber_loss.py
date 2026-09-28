import numpy as np


def huber_loss(
    y_true: list,
    y_pred: list,
    delta: float = 1.0
) -> float:
    """
    Calculate the mean Huber loss.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        delta: Threshold between quadratic and linear regions.

    Returns:
        Mean Huber loss as a Python float.

    Formula:
        0.5 * e^2,                  if |e| <= delta
        delta * (|e| - 0.5 * delta), otherwise

        where e = y_true - y_pred.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = np.abs(y_true - y_pred)

    loss = np.where(
        error <= delta,
        0.5 * error ** 2,
        delta * (error - 0.5 * delta)
    )

    return float(np.mean(loss))
import numpy as np


def log_loss(
    y_true: list,
    y_pred: list,
    eps: float = 1e-15
):
    """
    Compute binary log loss independently for each sample.

    Args:
        y_true: Binary target labels.
        y_pred: Predicted probabilities.
        eps: Small value used to clip probabilities.

    Returns:
        A list of per-sample log loss values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    p = np.clip(y_pred, eps, 1 - eps)

    loss = -(
        y_true * np.log(p)
        + (1 - y_true) * np.log(1 - p)
    )

    return loss
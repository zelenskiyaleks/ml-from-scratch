import numpy as np


def hinge_loss(
    y_true: list,
    y_score: list,
    margin: float = 1.0,
    reduction: str = "mean"
) -> float:
    """
    Calculate binary hinge loss.

    Args:
        y_true: Binary labels in {-1, +1}.
        y_score: Prediction scores.
        margin: Desired classification margin.
        reduction: "mean" or "sum".

    Returns:
        Hinge loss as a Python float.

    Formula:
        loss = max(0, margin - y_true * y_score)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    losses = np.maximum(0, margin - y_true * y_score)

    if reduction == "mean":
        return float(np.mean(losses))

    if reduction == "sum":
        return float(np.sum(losses))

    raise ValueError("reduction must be 'mean' or 'sum'")
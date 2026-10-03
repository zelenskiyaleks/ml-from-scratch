import numpy as np


def linear_warmup(
    X: list,
    y: list,
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int
) -> dict:
    """
    Train linear regression with a linear learning-rate warmup.

    Args:
        X: Feature matrix.
        y: Target values.
        base_lr: Maximum learning rate.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of training epochs.

    Returns:
        Dictionary containing the learning-rate schedule
        and MSE loss curve.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    res = {}
    w = np.zeros(X.shape[1])
    loss = []
    lr_schedule = []

    for epoch in range(total_epochs):
        loss_i = ((y - X @ w) ** 2).mean()
        loss.append(float(loss_i))

        gr = 2 / X.shape[0] * X.T @ (X @ w - y)

        lr = base_lr * min(1, (epoch + 1) / warmup_epochs)
        lr_schedule.append(lr)

        w -= lr * gr

    res["lr_schedule"] = lr_schedule
    res["losses"] = loss

    return res
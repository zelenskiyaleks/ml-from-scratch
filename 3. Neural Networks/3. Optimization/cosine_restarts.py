import numpy as np


def cosine_restarts(
    X: list,
    y: list,
    eta_max: float,
    eta_min: float,
    T_0: int,
    T_mult: int,
    total_epochs: int
) -> dict:
    """
    Train linear regression with cosine annealing and warm restarts.

    Args:
        X: Feature matrix.
        y: Target values.
        eta_max: Maximum learning rate.
        eta_min: Minimum learning rate.
        T_0: Initial cycle length.
        T_mult: Factor used to increase the cycle length.
        total_epochs: Total number of training epochs.

    Returns:
        Dictionary containing the learning-rate schedule
        and pre-update MSE losses.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    res = {}
    w = np.zeros(X.shape[1])
    loss = []
    lr_schedule = []

    T_i = T_0
    t_cum = 0

    for iteration in range(total_epochs):
        loss_i = ((y - X @ w) ** 2).mean()
        loss.append(float(loss_i))

        gr = 2 / X.shape[0] * X.T @ (X @ w - y)

        T_cur = iteration - t_cum

        lr = eta_min + (eta_max - eta_min) / 2 * (
            1 + np.cos(np.pi * T_cur / T_i)
        )

        lr_schedule.append(float(lr))
        w -= lr * gr

        if T_cur + 1 >= T_i:
            t_cum = iteration + 1
            T_i *= T_mult

    res["lr_schedule"] = lr_schedule
    res["losses"] = loss

    return res
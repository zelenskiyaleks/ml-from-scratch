import numpy as np


def nesterov_momentum(
    X: list,
    y: list,
    lr: float,
    beta: float,
    n_epochs: int
) -> dict:
    """
    Compare classical momentum and Nesterov momentum
    for linear regression using mean squared error.

    Args:
        X: Feature matrix.
        y: Target values.
        lr: Learning rate.
        beta: Momentum coefficient.
        n_epochs: Number of training epochs.

    Returns:
        Dictionary containing classical_losses and
        nesterov_losses.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    res = {}

    # Classical Momentum
    w = np.zeros(X.shape[1])
    v_t = np.zeros(X.shape[1])
    loss = []

    for _ in range(n_epochs):
        loss_i = ((y - X @ w) ** 2).mean()
        loss.append(float(loss_i))

        grad = 2 / X.shape[0] * X.T @ (X @ w - y)
        v_t = beta * v_t + grad
        w -= lr * v_t

    res["classical_losses"] = loss

    # Nesterov Momentum
    w = np.zeros(X.shape[1])
    v_t = np.zeros(X.shape[1])
    loss = []

    for _ in range(n_epochs):
        loss_i = ((y - X @ w) ** 2).mean()
        loss.append(float(loss_i))

        w_look = w - lr * beta * v_t
        grad = 2 / X.shape[0] * X.T @ (X @ w_look - y)

        v_t = beta * v_t + grad
        w -= lr * v_t

    res["nesterov_losses"] = loss

    return res
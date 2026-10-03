import numpy as np


def gradient_clipping(
    X: list,
    y: list,
    lr: float,
    clip_value: float,
    clip_norm: float,
    n_epochs: int
) -> dict:
    """
    Compare gradient descent with no clipping, value clipping,
    and global norm clipping.

    Args:
        X: Input feature matrix.
        y: Target values.
        lr: Learning rate.
        clip_value: Maximum absolute gradient value.
        clip_norm: Maximum global gradient L2 norm.
        n_epochs: Number of training epochs.

    Returns:
        Dictionary containing MSE loss curves for all three methods.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    res = {}

    # Vanilla gradient descent
    w = np.zeros(X.shape[1])
    b = 0
    loss = []

    for _ in range(n_epochs):
        residual = X @ w + b - y

        gr_w = 2 / X.shape[0] * (X.T @ residual)
        gr_b = 2 / X.shape[0] * np.sum(residual)

        w -= lr * gr_w
        b -= lr * gr_b

        loss_i = np.mean((y - X @ w - b) ** 2)
        loss.append(float(loss_i))

    res["unclipped_losses"] = loss

    # Value clipping
    w = np.zeros(X.shape[1])
    b = 0
    loss = []

    for _ in range(n_epochs):
        residual = X @ w + b - y

        gr_w = 2 / X.shape[0] * (X.T @ residual)
        gr_b = 2 / X.shape[0] * np.sum(residual)

        gr_w = np.clip(gr_w, -clip_value, clip_value)
        gr_b = np.clip(gr_b, -clip_value, clip_value)

        w -= lr * gr_w
        b -= lr * gr_b

        loss_i = np.mean((y - X @ w - b) ** 2)
        loss.append(float(loss_i))

    res["clip_value_losses"] = loss

    # Global norm clipping
    w = np.zeros(X.shape[1])
    b = 0
    loss = []

    for _ in range(n_epochs):
        residual = X @ w + b - y

        gr_w = 2 / X.shape[0] * (X.T @ residual)
        gr_b = 2 / X.shape[0] * np.sum(residual)

        grad_norm = np.sqrt(np.sum(gr_w ** 2) + gr_b ** 2)
        clip_coef = min(1.0, clip_norm / grad_norm)

        gr_w *= clip_coef
        gr_b *= clip_coef

        w -= lr * gr_w
        b -= lr * gr_b

        loss_i = np.mean((y - X @ w - b) ** 2)
        loss.append(float(loss_i))

    res["clip_norm_losses"] = loss

    return res
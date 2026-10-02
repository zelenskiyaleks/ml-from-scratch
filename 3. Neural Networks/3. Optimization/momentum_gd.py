import numpy as np


def momentum_gd(
    X: list,
    y: list,
    lr: float,
    beta: float,
    n_epochs: int
) -> dict:
    """
    Compare vanilla gradient descent with classical momentum.

    Args:
        X: Feature matrix.
        y: Target values.
        lr: Learning rate.
        beta: Momentum coefficient.
        n_epochs: Number of training epochs.

    Returns:
        A dictionary containing vanilla_losses and momentum_losses.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape

    # Vanilla Gradient Descent
    w = np.zeros(n_features)
    vanilla_losses = []

    for _ in range(n_epochs):
        loss = np.mean((X @ w - y) ** 2)
        vanilla_losses.append(float(loss))

        grad = (2 / n_samples) * X.T @ (X @ w - y)
        w -= lr * grad

    # Classical Momentum
    w = np.zeros(n_features)
    velocity = np.zeros(n_features)
    momentum_losses = []

    for _ in range(n_epochs):
        loss = np.mean((X @ w - y) ** 2)
        momentum_losses.append(float(loss))

        grad = (2 / n_samples) * X.T @ (X @ w - y)

        velocity = beta * velocity + grad
        w -= lr * velocity

    return {
        "vanilla_losses": vanilla_losses,
        "momentum_losses": momentum_losses
    }
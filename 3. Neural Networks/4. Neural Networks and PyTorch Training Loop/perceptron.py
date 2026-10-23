import numpy as np


def perceptron(
    X: list,
    y: list,
    lr: float = 0.1,
    epochs: int = 100
) -> tuple:
    """
    Train a binary perceptron.

    Args:
        X: Input feature matrix.
        y: Binary target labels.
        lr: Learning rate.
        epochs: Number of training epochs.

    Returns:
        Tuple containing the weight list and bias,
        rounded to four decimal places.

    Update rules:
        weights = weights + lr * error * x
        bias = bias + lr * error
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    weights = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0

    for _ in range(epochs):
        for x, target in zip(X, y):
            z = weights @ x + bias
            prediction = 1.0 if z >= 0 else 0.0
            error = target - prediction

            weights += lr * error * x
            bias += lr * error

    return np.round(weights, 4).tolist(), round(float(bias), 4)
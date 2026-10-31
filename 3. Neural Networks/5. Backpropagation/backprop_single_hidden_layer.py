import numpy as np


def backprop(
    X: list,
    y: list,
    W1: list,
    b1: list,
    W2: list,
    b2: list
) -> dict:
    """
    Compute gradients through a two-layer neural network.

    Args:
        X: Input batch.
        y: Target values.
        W1: First layer weight matrix.
        b1: First layer bias vector.
        W2: Second layer weight matrix.
        b2: Second layer bias vector.

    Returns:
        Dictionary containing gradients for both layers,
        rounded to six decimal places.

    Formulas:
        Z1 = X @ W1 + b1
        A1 = ReLU(Z1)
        Y_pred = A1 @ W2 + b2
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)

    # Forward pass
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    y_pred = A1 @ W2 + b2

    # Output gradients
    n = X.shape[0]
    dY = (-2 / n) * (y - y_pred)

    dW2 = A1.T @ dY
    db2 = np.sum(dY, axis=0)

    # Backpropagation through the second layer
    dA1 = dY @ W2.T

    # Backpropagation through ReLU
    dZ1 = dA1 * (Z1 > 0)

    # Gradients for the first layer
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {
        "dW1": np.round(dW1, 6).tolist(),
        "db1": np.round(db1, 6).tolist(),
        "dW2": np.round(dW2, 6).tolist(),
        "db2": np.round(db2, 6).tolist()
    }
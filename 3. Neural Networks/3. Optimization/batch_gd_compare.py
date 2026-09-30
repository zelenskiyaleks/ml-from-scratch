import numpy as np


def batch_gd_compare(
    X: list,
    y: list,
    batch_sizes: list,
    n_epochs: int,
    lr: float,
    seed: int
) -> list:
    """
    Compare gradient descent with different batch sizes.

    Args:
        X: Feature matrix.
        y: Target values.
        batch_sizes: List of batch sizes to compare.
        n_epochs: Number of training epochs.
        lr: Learning rate.
        seed: Random seed used for shuffling.

    Returns:
        One full-dataset MSE loss curve per batch size.

    Formula:
        g = (2 / B) * X_B.T @ (X_B @ w - y_B)

        w = w - lr * g
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    results = []

    for batch_size in batch_sizes:
        weights = np.zeros(X.shape[1])
        rng = np.random.RandomState(seed)

        loss_curve = []

        for _ in range(n_epochs):
            indices = rng.permutation(len(X))

            for start in range(0, len(X), batch_size):
                batch_indices = indices[start:start + batch_size]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                gradient = (
                    2 / len(X_batch)
                    * X_batch.T
                    @ (X_batch @ weights - y_batch)
                )

                weights -= lr * gradient

            loss = np.mean((X @ weights - y) ** 2)
            loss_curve.append(float(loss))

        results.append(loss_curve)

    return results
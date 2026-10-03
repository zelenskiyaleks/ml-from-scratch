import numpy as np


def onecycle_train(
    X: list,
    y: list,
    max_lr: float,
    div_factor: float,
    final_div: float,
    pct_start: float,
    total_epochs: int
) -> dict:
    """
    Train a linear regression model using the One-Cycle learning-rate policy.

    The learning rate follows two phases:
    1. Warmup: cosine increase from initial_lr to max_lr.
    2. Decay: cosine decrease from max_lr to min_lr.

    Args:
        X: Feature matrix.
        y: Target values.
        max_lr: Maximum learning rate.
        div_factor: Factor used to calculate the initial learning rate.
        final_div: Factor used to calculate the minimum learning rate.
        pct_start: Fraction of training allocated to warmup.
        total_epochs: Total number of training epochs.

    Returns:
        A dictionary containing:
            - lr_schedule: Learning rate at each epoch.
            - losses: MSE before each weight update.
            - initial_lr: Initial learning rate.
            - min_lr: Minimum learning rate.

    Formulas:
        initial_lr = max_lr / div_factor
        min_lr = initial_lr / final_div
    """
    X = np.asarray(X)
    y = np.asarray(y)

    w = np.zeros(X.shape[1])

    losses = []
    lr_schedule = []

    initial_lr = max_lr / div_factor
    min_lr = initial_lr / final_div

    peak_index = int(total_epochs * pct_start)
    peak_index = max(1, min(peak_index, total_epochs - 2))

    for epoch in range(total_epochs):
        # Calculate MSE before the weight update
        y_pred = X @ w
        loss = np.mean((y - y_pred) ** 2)
        losses.append(float(loss))

        # Calculate gradient
        grad = (2 / X.shape[0]) * X.T @ (y_pred - y)

        # Warmup phase
        if epoch <= peak_index:
            lr = initial_lr + (max_lr - initial_lr) / 2 * (
                1 - np.cos(np.pi * epoch / peak_index)
            )

        # Cosine decay phase
        else:
            lr = min_lr + (max_lr - min_lr) / 2 * (
                1 + np.cos(
                    np.pi * (epoch - peak_index) /
                    (total_epochs - 1 - peak_index)
                )
            )

        lr_schedule.append(float(lr))

        # Update weights
        w -= lr * grad

    return {
        "lr_schedule": lr_schedule,
        "losses": losses,
        "initial_lr": float(initial_lr),
        "min_lr": float(min_lr)
    }
import torch
import torch.nn as nn


def manual_train_step(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    lr: float
) -> float:
    """
    Perform one manual gradient descent training step.

    Args:
        model: PyTorch model.
        X: Input features.
        y: True target values.
        criterion: Loss function.
        lr: Learning rate.

    Returns:
        Batch loss before the parameter update as a Python float.

    Update rule:
        parameter = parameter - lr * gradient
    """
    model.train()
    model.zero_grad()

    # Forward pass
    y_pred = model(X)

    # Compute loss before parameter update
    loss = criterion(y, y_pred)

    # Backward pass
    loss.backward()

    # Manual gradient descent update
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

    # Clear gradients
    model.zero_grad()

    return float(loss.item())
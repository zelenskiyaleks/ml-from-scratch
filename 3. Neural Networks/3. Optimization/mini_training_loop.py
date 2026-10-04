import torch
import torch.nn as nn


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer
) -> float:
    """
    Train a model for one epoch using mini-batch gradient descent.

    Args:
        model: PyTorch model to train.
        dataloader: DataLoader providing batches of input data and targets.
        criterion: Loss function.
        optimizer: Optimizer used to update model parameters.

    Returns:
        Mean batch loss as a Python float.

    Training steps:
        1. Set the model to training mode.
        2. Iterate over mini-batches.
        3. Reset gradients.
        4. Perform forward pass.
        5. Calculate loss.
        6. Perform backward pass.
        7. Update model parameters.
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in dataloader:
        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_batch)

        # Calculate loss
        loss = criterion(y_pred, y_batch)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
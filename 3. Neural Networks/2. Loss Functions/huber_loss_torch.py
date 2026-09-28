import torch


def huber_loss(y_true, y_pred, delta=1.0):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)

    error = torch.abs(y_true - y_pred)

    loss = torch.where(
        error <= delta,
        0.5 * error ** 2,
        delta * (error - 0.5 * delta)
    )

    return torch.mean(loss)
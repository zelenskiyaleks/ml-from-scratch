import torch


def hinge_loss(
    y_true,
    y_score,
    margin=1.0,
    reduction="mean"
):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_score = torch.tensor(y_score, dtype=torch.float32)

    losses = torch.clamp(
        margin - y_true * y_score,
        min=0
    )

    if reduction == "mean":
        return torch.mean(losses)

    if reduction == "sum":
        return torch.sum(losses)

    raise ValueError("reduction must be 'mean' or 'sum'")
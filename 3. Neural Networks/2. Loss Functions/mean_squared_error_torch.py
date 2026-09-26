import torch


def mean_squared_error(y_pred, y_true):
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)

    return torch.mean((y_pred - y_true) ** 2)
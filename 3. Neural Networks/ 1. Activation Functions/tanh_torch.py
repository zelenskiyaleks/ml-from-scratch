import torch


def tanh(x):
    x = torch.tensor(x)

    return (torch.exp(x) - torch.exp(-x)) / (
        torch.exp(x) + torch.exp(-x)
    )
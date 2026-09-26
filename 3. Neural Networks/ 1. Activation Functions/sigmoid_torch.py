import torch

def sigmoid(x):

    x = torch.tensor(x)

    return 1 / (1 + torch.exp(-x))
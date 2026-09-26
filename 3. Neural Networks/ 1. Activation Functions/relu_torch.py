import torch

def relu(x):
    x = torch.tensor(x)
    x = torch.maximum(x, torch.tensor(0))
    
    return x
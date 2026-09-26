import torch


def softmax(x):
    x = torch.tensor(x)

    max_per_row = torch.max(x, dim=-1, keepdim=True).values
    
    exp_x = torch.exp(x - max_per_row)
    
    return exp_x / torch.sum(exp_x, axis=-1, keepdims=True)
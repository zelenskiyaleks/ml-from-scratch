import numpy as np


def log_loss(y_true, y_pred, eps =  1e-15):

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    
    p = torch.clip(y_pred, eps, 1 - eps)
    
    loss = -(
        y_true * torch.log(p)
        + (1 - y_true) * torch.log(1 - p)
    )
    return loss
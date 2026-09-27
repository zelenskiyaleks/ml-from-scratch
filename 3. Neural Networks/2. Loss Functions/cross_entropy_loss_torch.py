import torch


def cross_entropy_loss(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    samples = torch.arange(len(y_true))
    true_class_probabilities = y_pred[samples, y_true]

    return -torch.log(true_class_probabilities).mean()
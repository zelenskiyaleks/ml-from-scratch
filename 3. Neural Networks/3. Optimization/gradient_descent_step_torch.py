import torch


def gradient_descent_step(values, gradients, learning_rate):
    values = torch.tensor(values)
    gradients = torch.tensor(gradients)
    values_next = values - learning_rate * gradients
    L = float((gradients * (values_next - values)).sum())
    return (values_next, L)
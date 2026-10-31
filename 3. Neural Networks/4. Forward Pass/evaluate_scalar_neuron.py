import torch


def neuron_forward(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor
) -> tuple:
    """
    Evaluate a scalar tanh neuron.

    Args:
        inputs: Input feature tensor.
        weights: Weight tensor aligned with inputs.
        bias: Scalar bias tensor.

    Returns:
        Tuple containing the scalar preactivation and tanh output.

    Formulas:
        a = sum(inputs * weights) + bias
        y = tanh(a)
    """
    # Compute weighted sum and add bias
    a = inputs @ weights + bias

    # Apply tanh activation
    y = torch.tanh(a)

    return a, y
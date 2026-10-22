import torch


def scalar_neuron_module(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    nonlinear: bool
) -> torch.Tensor:
    """
    Evaluate a scalar neuron with an optional tanh activation.

    Args:
        inputs: Input feature tensor.
        weights: Weight tensor aligned with inputs.
        bias: Scalar bias tensor.
        nonlinear: Whether to apply tanh activation.

    Returns:
        A scalar tensor preserving the input dtype and device.

    Formula:
        a = inputs @ weights + bias
        output = tanh(a) if nonlinear else a
    """
    # Compute weighted sum and add bias
    a = inputs @ weights + bias

    # Apply activation if requested
    if nonlinear:
        return torch.tanh(a)

    return a
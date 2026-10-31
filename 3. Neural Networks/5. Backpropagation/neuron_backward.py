import torch


def neuron_backward(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    upstream_gradient: torch.Tensor
) -> tuple:
    """
    Compute the forward output and backward gradients of a tanh neuron.

    Args:
        inputs: Input feature tensor.
        weights: Weight tensor aligned with inputs.
        bias: Scalar bias tensor.
        upstream_gradient: Gradient of the loss with respect to the output.

    Returns:
        Tuple containing:
            - Neuron output.
            - Gradients with respect to inputs.
            - Gradients with respect to weights.
            - Gradient with respect to bias.

    Formulas:
        a = inputs @ weights + bias
        y = tanh(a)
        delta = upstream_gradient * (1 - y ** 2)
        input_gradients = delta * weights
        weight_gradients = delta * inputs
        bias_gradient = delta
    """
    a = inputs @ weights + bias
    output = torch.tanh(a)

    delta = upstream_gradient * (1 - output ** 2)

    input_gradients = delta * weights
    weight_gradients = delta * inputs
    bias_gradient = delta

    return (
        output,
        input_gradients,
        weight_gradients,
        bias_gradient
    )
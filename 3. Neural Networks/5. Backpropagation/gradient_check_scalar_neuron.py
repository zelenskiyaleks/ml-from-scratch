import torch

def forward(inputs, weights, bias):
     """
    Compute the output of a scalar tanh neuron.

    Args:
        inputs: One-dimensional tensor containing input features.
        weights: One-dimensional tensor containing weights.
        bias: Scalar bias tensor.

    Returns:
        Scalar output of the neuron.

    Formula:
        z = sum(inputs * weights) + bias
        y = tanh(z)
    """
    return torch.tanh(torch.sum(inputs * weights) + bias)

def neuron_gradient_check(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, h: float) -> tuple:
    """
    Compare analytical and numerical gradients of a scalar tanh neuron.

    The function computes gradients with respect to weights and bias
    using two approaches:

    1. Analytical gradients derived manually using the chain rule.
    2. Numerical gradients estimated using forward finite differences.

    All calculations are performed using PyTorch float64 tensors.
    Automatic differentiation is not used.

    Args:
        inputs: One-dimensional input tensor.
        weights: One-dimensional weight tensor of the same length as inputs.
        bias: Scalar bias tensor.
        h: Positive perturbation step used for numerical differentiation.

    Returns:
        Tuple containing:
            - Analytical gradients with respect to weights.
            - Numerical gradients with respect to weights.
            - Analytical gradient with respect to bias.
            - Numerical gradient with respect to bias.
            - Maximum absolute difference between analytical and
              numerical gradients.

    Formulas:
        z = sum(w_i * x_i) + b
        y = tanh(z)

        Analytical gradients:
            dy/dw_i = (1 - y^2) * x_i
            dy/db   = 1 - y^2

        Numerical gradients using forward differences:
            dy/dw_i ≈ (y(w_i + h) - y(w_i)) / h
            dy/db   ≈ (y(b + h) - y(b)) / h

        Maximum absolute error:
            max_error = max(
                |analytical_gradients_w - numerical_gradients_w|,
                |analytical_gradient_b - numerical_gradient_b|
            )
    """
    inputs = inputs.to(dtype=torch.float64)
    weights = weights.to(dtype=torch.float64)
    bias = bias.to(dtype=torch.float64)

    y = forward(inputs, weights, bias)

    analytical_gradients_w = (1 - y**2) * inputs
    analytical_gradient_b = 1 - y**2

    numerical_gradients_w = []

    y_original = y

    numerical_gradients_w = torch.empty_like(weights)

    for i in range(len(weights)):
        weights_plus = weights.clone()
        weights_plus[i] += h
    
        y_plus = forward(inputs, weights_plus, bias)
    
        numerical_gradients_w[i] = (y_plus - y_original) / h

    y_plus = forward(inputs, weights, bias + h)
    numerical_gradient_b = (y_plus - y_original) / h

    error_w = torch.abs(
        analytical_gradients_w - numerical_gradients_w
    )

    error_b = torch.abs(
        analytical_gradient_b - numerical_gradient_b
    )

    max_error = torch.max(
        torch.cat([error_w, error_b.reshape(1)])
    )

    return (
        analytical_gradients_w,
        numerical_gradients_w,
        analytical_gradient_b,
        numerical_gradient_b,
        max_error
    )
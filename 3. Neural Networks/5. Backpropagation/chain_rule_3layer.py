import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(w1, w2, w3, x):
    z1 = w1 * x
    a1 = sigmoid(z1)

    z2 = w2 * a1
    a2 = sigmoid(z2)

    z3 = w3 * a2
    return sigmoid(z3)


def chain_rule_3layer(
    w1: float,
    w2: float,
    w3: float,
    x: float
) -> dict:
    """
    Compute the analytical and numerical gradients of a three-layer
    sigmoid neural network with respect to the first weight.

    Args:
        w1: Weight connecting the input to the first neuron.
        w2: Weight connecting the first neuron to the second neuron.
        w3: Weight connecting the second neuron to the output neuron.
        x: Scalar input value.

    Returns:
        Dictionary containing:
            - factors: Six local derivatives used in the chain rule.
            - analytical_gradient: Gradient computed analytically.
            - numerical_gradient: Gradient estimated using central difference.

    Network architecture:
        z1 = w1 * x
        a1 = sigmoid(z1)

        z2 = w2 * a1
        a2 = sigmoid(z2)

        z3 = w3 * a2
        y = sigmoid(z3)

    Formulas:
        sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))

        dy/dw1 = sigmoid'(z3) * w3
                 * sigmoid'(z2) * w2
                 * sigmoid'(z1) * x

        The analytical gradient is calculated by multiplying
        the six local derivatives in the specified order.

        The numerical gradient is estimated using central difference:
            (f(w1 + h) - f(w1 - h)) / (2 * h)

        where h = 0.00001.
    """
    h = 0.00001

    # Forward pass
    z1 = w1 * x
    a1 = sigmoid(z1)

    z2 = w2 * a1
    a2 = sigmoid(z2)

    z3 = w3 * a2
    y = sigmoid(z3)

    # Chain rule factors
    factors = [
        y * (1 - y),
        w3,
        a2 * (1 - a2),
        w2,
        a1 * (1 - a1),
        x
    ]

    # Analytical gradient
    analytical_gradient = np.prod(factors)

    # Numerical gradient: central difference
    y_plus = forward(w1 + h, w2, w3, x)
    y_minus = forward(w1 - h, w2, w3, x)

    numerical_gradient = (y_plus - y_minus) / (2 * h)

    return {
        "factors": [float(f) for f in factors],
        "analytical_gradient": float(analytical_gradient),
        "numerical_gradient": float(numerical_gradient)
    }
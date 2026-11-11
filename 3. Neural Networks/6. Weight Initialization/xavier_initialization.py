import math


def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Apply Xavier (Glorot) uniform initialization to a weight matrix.

    Xavier initialization scales weights according to the number of
    input and output connections to maintain a stable variance of
    activations and gradients across neural network layers.

    Args:
        W: Weight matrix with values in the range [0, 1].
        fan_in: Number of input units in the layer.
        fan_out: Number of output units in the layer.

    Returns:
        Weight matrix mapped to the Xavier uniform range [-L, L],
        preserving the original matrix shape.

    Formulas:
        L = sqrt(6 / (fan_in + fan_out))

        W_new = W * 2 * L - L
    """
    L = math.sqrt(6 / (fan_in + fan_out))

    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] = W[i][j] * 2 * L - L

    return W

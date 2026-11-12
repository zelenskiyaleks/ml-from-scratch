import math


def he_initialization(W: list, fan_in: int) -> list:
    """
    Apply He (Kaiming) uniform initialization to a weight matrix.

    He initialization is designed for neural networks using ReLU
    activation functions. It compensates for the reduction in
    activation variance caused by ReLU zeroing negative values.

    Args:
        W: Weight matrix with values in the range [0, 1].
        fan_in: Number of input units in the layer.

    Returns:
        Weight matrix mapped to the He uniform range [-L, L],
        preserving the original matrix shape.

    Formulas:
        L = sqrt(6 / fan_in)

        W_new = W * 2 * L - L
    """
    L = math.sqrt(6 / fan_in)

    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] = round(W[i][j] * 2 * L - L, 4)

    return W

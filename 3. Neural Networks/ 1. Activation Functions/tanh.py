import numpy as np


def tanh(x: list) -> np.ndarray:
    """
    Compute the hyperbolic tangent activation function.

    Args:
        x: Input values.

    Returns:
        A NumPy array with the same shape as the input.

    Formula:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """

    x = np.asarray(x, dtype=float)

    return (np.exp(x) - np.exp(-x)) / (
        np.exp(x) + np.exp(-x)
    )
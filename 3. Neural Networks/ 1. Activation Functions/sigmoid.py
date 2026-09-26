import numpy as np


def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Compute the sigmoid activation function.

    Args:
        x: A scalar, Python list, or nested Python list.

    Returns:
        A float for a scalar input, or a NumPy array
        with the same shape as the input.

    Formula:
        sigmoid(x) = 1 / (1 + exp(-x))
    """

    x = np.asarray(x)

    return 1 / (1 + np.exp(-x))
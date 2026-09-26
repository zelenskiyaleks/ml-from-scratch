import numpy as np


def relu(x: list | float) -> np.ndarray:
    """
    Compute the ReLU activation function.

    Args:
        x: A scalar, Python list, or nested Python list.

    Returns:
        A NumPy array with the same shape as the input.

    Formula:
        ReLU(x) = max(0, x)
    """

    x = np.asarray(x)

    return np.maximum(x, 0)
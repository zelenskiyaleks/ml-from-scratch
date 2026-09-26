import numpy as np


def softmax(x: list) -> np.ndarray:
    """
    Compute numerically stable softmax probabilities.

    For a 1D input, normalize the whole vector.
    For a 2D input, normalize each row independently.

    Args:
        x: A 1D or 2D list of logits.

    Returns:
        A NumPy array with the same shape as the input.

    Formula:
        softmax(x_i) = exp(x_i - max(x)) /
                       sum(exp(x_j - max(x)))
    """

    x = np.asarray(x, dtype=float)

    max_per_row = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_per_row)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
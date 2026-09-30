import numpy as np


def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8
) -> tuple[list, list]:
    """
    Apply one RMSProp optimization step.

    Args:
        w: Current parameter values.
        g: Current gradients.
        s: Running average of squared gradients.
        lr: Learning rate.
        beta: Decay factor for the running average.
        eps: Small value for numerical stability.

    Returns:
        A tuple containing updated parameters and
        the updated running average.

    Formulas:
        s_t = beta * s_{t-1} + (1 - beta) * g_t^2

        w_t = w_{t-1} -
              lr * g_t / sqrt(s_t + eps)
    """
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    s_t = beta * s + (1 - beta) * g ** 2

    w_t = w - lr * g / np.sqrt(s_t + eps)

    return w_t, s_t
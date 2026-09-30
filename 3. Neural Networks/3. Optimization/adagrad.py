import numpy as np


def adagrad_step(
    w: list,
    g: list,
    G: list,
    lr: float = 0.01,
    eps: float = 1e-8
) -> dict:
    """
    Apply one AdaGrad optimization step.

    Args:
        w: Current parameter values.
        g: Current gradients.
        G: Accumulated squared gradients.
        lr: Learning rate.
        eps: Small value for numerical stability.

    Returns:
        Dictionary containing updated parameters and
        accumulated squared gradients.

    Formulas:
        G_t = G_{t-1} + g_t^2

        w_t = w_{t-1} -
              lr * g_t / (sqrt(G_t) + eps)
    """
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    G = np.asarray(G, dtype=float)

    G_t = G + g ** 2
    w_t = w - lr * g / (np.sqrt(G_t) + eps)

    return {
        "new_w": w_t,
        "new_G": G_t
    }
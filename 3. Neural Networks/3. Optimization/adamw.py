
import numpy as np


def adamw_step(
    w: list,
    m: list,
    v: list,
    grad: list,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    eps: float = 1e-8
) -> dict:
    """
    Perform one AdamW optimizer update step.

    Args:
        w: Current model parameters.
        m: First moment estimates.
        v: Second moment estimates.
        grad: Current gradients.
        lr: Learning rate.
        beta1: First moment decay factor.
        beta2: Second moment decay factor.
        weight_decay: Decoupled weight decay coefficient.
        eps: Small constant for numerical stability.

    Returns:
        A dictionary containing updated parameters (new_w),
        first moments (new_m), and second moments (new_v).

    Formulas:
        m_t = beta1 * m + (1 - beta1) * grad
        v_t = beta2 * v + (1 - beta2) * grad^2

        w_t = w - lr * m_t / (sqrt(v_t) + eps)
               - lr * weight_decay * w
    """
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * grad ** 2

    new_w = (
        w
        - lr * new_m / (np.sqrt(new_v) + eps)
        - lr * weight_decay * w
    )

    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v
    }
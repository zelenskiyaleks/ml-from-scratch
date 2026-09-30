import numpy as np


def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply one Adam optimization step.

    Args:
        param: Current parameter values.
        grad: Current gradients.
        m: First-moment estimate.
        v: Second-moment estimate.
        t: One-based timestep.
        lr: Learning rate.
        beta1: Decay rate for the first moment.
        beta2: Decay rate for the second moment.
        eps: Small value for numerical stability.

    Returns:
        A tuple containing updated parameters, first moment,
        and second moment.

    Formulas:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

        m_hat_t = m_t / (1 - beta1^t)
        v_hat_t = v_t / (1 - beta2^t)

        param_t = param_{t-1} -
                  lr * m_hat_t / (sqrt(v_hat_t) + eps)
    """
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    m_t = beta1 * m + (1 - beta1) * grad
    v_t = beta2 * v + (1 - beta2) * grad ** 2

    m_hat = m_t / (1 - beta1 ** t)
    v_hat = v_t / (1 - beta2 ** t)

    param_new = (
        param
        - lr * m_hat / (np.sqrt(v_hat) + eps)
    )

    return (param_new, m_t, v_t)
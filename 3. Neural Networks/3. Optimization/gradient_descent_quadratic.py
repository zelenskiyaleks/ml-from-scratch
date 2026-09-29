def gradient_descent_quadratic(
    a: float,
    b: float,
    c: float,
    x0: float,
    lr: float,
    steps: int
) -> float:
    """
    Minimize a 1D quadratic function using gradient descent.

    Args:
        a: Quadratic coefficient.
        b: Linear coefficient.
        c: Constant coefficient.
        x0: Initial value of x.
        lr: Learning rate.
        steps: Number of gradient descent steps.

    Returns:
        Final value of x after the requested number of iterations.

    Formula:
        f(x) = ax^2 + bx + c
        f'(x) = 2ax + b
        x = x - lr * f'(x)
    """
    x = x0

    for _ in range(steps):
        gradient = 2 * a * x + b
        x -= lr * gradient

    return float(x)
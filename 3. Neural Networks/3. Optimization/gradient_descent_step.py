import numpy as np


def gradient_descent_step(
    values: list,
    gradients: list,
    learning_rate: float
) -> tuple[list, float]:
    """
    Apply one gradient descent update step.

    Args:
        values: Current parameter values.
        gradients: Gradients of the objective with respect to the parameters.
        learning_rate: Step size.

    Returns:
        A tuple containing:
        - updated parameter values as a list of Python floats;
        - predicted first-order change in the objective.

    Formula:
        theta_new = theta - learning_rate * gradient

        Delta L_pred = sum(
            gradient * (theta_new - theta)
        )
    """
    values = np.asarray(values, dtype=float)
    gradients = np.asarray(gradients, dtype=float)

    values_next = values - learning_rate * gradients

    predicted_change = float(
        np.sum(gradients * (values_next - values))
    )

    return (values_next.tolist(), predicted_change)
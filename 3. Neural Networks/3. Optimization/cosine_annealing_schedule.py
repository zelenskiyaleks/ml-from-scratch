import math


def cosine_annealing_schedule(
    base_lr: float,
    min_lr: float,
    total_steps: int,
    current_step: int
) -> float:
    """
    Calculate the learning rate using cosine annealing.

    Args:
        base_lr: Initial maximum learning rate.
        min_lr: Minimum learning rate.
        total_steps: Total number of training steps.
        current_step: Current training step.

    Returns:
        Learning rate for the current step.

    Formula:
        lr = min_lr + 0.5 * (base_lr - min_lr) *
             (1 + cos(pi * current_step / total_steps))
    """
    lr = min_lr + 0.5 * (base_lr - min_lr) * (
        1 + math.cos(math.pi * current_step / total_steps)
    )

    return lr
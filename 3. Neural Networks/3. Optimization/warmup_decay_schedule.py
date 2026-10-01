def warmup_decay_schedule(
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    current_step: int
) -> float:
    """
    Calculate the learning rate using linear warmup and decay.

    Args:
        base_lr: Initial maximum learning rate.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        current_step: Current training step.

    Returns:
        Learning rate for the current step.

    Formula:
        Warmup:
            lr = base_lr * current_step / warmup_steps

        Linear decay:
            lr = base_lr * (total_steps - current_step)
                 / (total_steps - warmup_steps)
    """
    if current_step < warmup_steps:
        return base_lr * current_step / warmup_steps

    return base_lr * (
        (total_steps - current_step)
        / (total_steps - warmup_steps)
    )
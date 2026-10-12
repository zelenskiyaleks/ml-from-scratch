
import torch


class CustomSGD(torch.optim.Optimizer):
    """
    Custom SGD optimizer with momentum.

    Args:
        params: Model parameters to optimize.
        lr: Learning rate.
        momentum: Momentum coefficient.

    Update rules:
        v_t = momentum * v_(t-1) + gradient
        parameter = parameter - lr * v_t
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum
        }

        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform one optimization step.

        Args:
            closure: Optional function that reevaluates the model
                and returns the loss.

        Returns:
            Loss returned by closure, or None.
        """
        loss = None

        # Optionally reevaluate loss and gradients
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update parameters without tracking gradients
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]

                for param in group["params"]:
                    if param.grad is None:
                        continue

                    state = self.state[param]

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param)

                    buffer = state["momentum_buffer"]
                    buffer.mul_(momentum).add_(param.grad)

                    param.add_(buffer, alpha=-lr)

        return loss


def backprop_graph(w1: float, w2: float, x: float) -> dict:
    """
    Perform forward and backward passes through a scalar
    computational graph.

    The graph computes the following function:

        prod1 = w1 * x
        x_sq = x ** 2
        prod2 = w2 * x_sq
        sum_val = prod1 + prod2
        output = ReLU(sum_val)

    Args:
        w1: First scalar weight.
        w2: Second scalar weight.
        x: Scalar input value.

    Returns:
        Dictionary containing two sections:

        forward:
            - prod1: Product of w1 and x.
            - x_sq: Squared input.
            - prod2: Product of w2 and x squared.
            - sum_val: Sum of both products before activation.
            - output: Final ReLU output.

        gradients:
            - d_w1: Derivative of output with respect to w1.
            - d_w2: Derivative of output with respect to w2.
            - d_x: Derivative of output with respect to x.

    Formulas:
        output = ReLU(w1 * x + w2 * x ** 2)

        ReLU'(z) = 1 if z > 0 else 0

        d_output / d_w1 = ReLU'(sum_val) * x
        d_output / d_w2 = ReLU'(sum_val) * x ** 2
        d_output / d_x = ReLU'(sum_val) * (w1 + 2 * w2 * x)

    Notes:
        - Intermediate values are stored during the forward pass.
        - Gradients are computed manually using the chain rule.
        - The ReLU derivative is defined as zero when its input is zero.
    """

    res = {}
    res["forward"] = {}

    # Forward pass
    res["forward"]["prod1"] = w1 * x
    res["forward"]["x_sq"] = x ** 2
    res["forward"]["prod2"] = w2 * res["forward"]["x_sq"]
    res["forward"]["sum_val"] = (
        res["forward"]["prod1"] + res["forward"]["prod2"]
    )

    if res["forward"]["sum_val"] < 0:
        res["forward"]["output"] = 0
    else:
        res["forward"]["output"] = res["forward"]["sum_val"]

    # Backward pass
    res["gradients"] = {}

    if res["forward"]["sum_val"] <= 0:
        relu = 0
    else:
        relu = 1

    res["gradients"]["d_w1"] = relu * x
    res["gradients"]["d_w2"] = relu * x ** 2
    res["gradients"]["d_x"] = relu * (w1 + 2 * w2 * x)

    return res

def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Compute the output of a linear layer for every input sample.

    Args:
        X: Input matrix of shape (n_samples, d_in).
        W: Weight matrix of shape (d_in, d_out).
        b: Bias vector of shape (d_out,).

    Returns:
        Output matrix of shape (n_samples, d_out).

    Formula:
        Y = X @ W + b
    """
    res = list()

    for i in range(len(X)):
        res_i = list()

        for j in range(len(W[0])):
            el = 0

            for k in range(len(X[0])):
                el += X[i][k] * W[k][j]

            el += b[j]
            res_i.append(el)

        res.append(res_i)

    return res
def autograd(operations: list, input_values: list) -> dict:
    """
    Build a scalar computational graph and compute gradients using
    reverse-mode automatic differentiation.

    Args:
        operations: List of operations describing the computational graph.
            Each operation is represented as a list:
                - ["add", a, b]: Addition of two parent nodes.
                - ["mul", a, b]: Multiplication of two parent nodes.
                - ["neg", a]: Negation of a parent node.

            Parent indices refer to nodes that already exist in the graph.
        input_values: List of initial scalar input values.
            Each input is assigned an index starting from zero.

    Returns:
        Dictionary containing:
            - output: Final scalar output of the computational graph.
            - gradients: List of gradients with respect to the initial
              input values, in their original order.

        All numeric outputs are rounded to four decimal places.

    Algorithm:
        Forward Pass:
            1. Initialize the graph with the input values.
            2. Process operations in their supplied order.
            3. Retrieve parent values using their node indices.
            4. Apply the corresponding operation (add, mul, or neg).
            5. Append each intermediate result to the values list.

        Backward Pass:
            1. Initialize all gradients with zero.
            2. Set the gradient of the final output node to one.
            3. Traverse operations in reverse order.
            4. Retrieve the gradient of the current operation's output.
            5. Apply the local derivative for each operation.
            6. Accumulate gradient contributions using += because
               a node can participate in multiple computational paths.

    Formulas:
        Addition:
            z = a + b
            dz/da = 1
            dz/db = 1

        Multiplication:
            z = a * b
            dz/da = b
            dz/db = a

        Negation:
            z = -a
            dz/da = -1
    """

    # Forward pass
    values = list(input_values)

    for op in operations:
        operation = op[0]
        parents = op[1:]

        if operation == "add":
            result = values[parents[0]] + values[parents[1]]

        elif operation == "mul":
            result = values[parents[0]] * values[parents[1]]

        elif operation == "neg":
            result = -values[parents[0]]

        values.append(result)

    # Backward pass
    grads = [0.0] * len(values)
    grads[-1] = 1.0

    for i in range(len(operations) - 1, -1, -1):
        op = operations[i]
        operation = op[0]
        parents = op[1:]

        child_idx = len(input_values) + i
        grad = grads[child_idx]

        if operation == "add":
            grads[parents[0]] += grad
            grads[parents[1]] += grad

        elif operation == "mul":
            grads[parents[0]] += grad * values[parents[1]]
            grads[parents[1]] += grad * values[parents[0]]

        elif operation == "neg":
            grads[parents[0]] -= grad

    return {
        "output": round(values[-1], 4),
        "gradients": [round(g, 4) for g in grads[:len(input_values)]]
    }
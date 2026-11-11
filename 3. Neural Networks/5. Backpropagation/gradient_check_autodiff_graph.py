
import numpy as np


def add(a, b):
    return a + b


def mul(a, b):
    return a * b


def tanh(z):
    return np.tanh(z)


# Local derivatives
def dmul(a, b):
    return b, a


def dsum(a, b):
    return 1.0, 1.0


def dtanh(output):
    return 1.0 - output**2


def autodiff_gradient_check(
    leaves: list,
    operations: list,
    output_id: str,
    h: float
) -> tuple:
    """
    Build a scalar computation graph and verify its analytical
    leaf gradients using forward finite differences.

    Args:
        leaves: List of dictionaries containing leaf node IDs
            and their initial scalar values.
        operations: List of operation dictionaries. Each operation
            defines its ID, operation type, and parent node IDs.
            Supported operations: add, mul, and tanh.
        output_id: ID of the graph node whose output is differentiated.
        h: Positive perturbation step for numerical gradient estimation.

    Returns:
        Tuple containing:
            - Dictionary of analytical gradients for reachable leaves.
            - Dictionary of numerical gradients for reachable leaves.
            - Maximum absolute difference between corresponding gradients.

    Algorithm:
        1. Build a mapping of operation IDs to operation definitions.
        2. Traverse the graph using DFS to obtain topological order.
        3. Perform a forward pass and store all intermediate values.
        4. Perform a reverse pass using the chain rule.
        5. Compute numerical gradients by perturbing each leaf separately.
        6. Calculate the maximum absolute difference between gradients.

    Local derivatives:
        add(a, b):
            da = 1
            db = 1

        mul(a, b):
            da = b
            db = a

        tanh(z):
            dz = 1 - tanh(z)^2

    Notes:
        - Only the selected output and its ancestors participate
          in gradient computation.
        - Intermediate values are stored for the backward pass.
        - Gradient contributions are accumulated when necessary.
        - Numerical gradients use forward finite differences.
        - All calculations use NumPy float64 arithmetic.
    """

    # 1. Prepare graph
    operations_dict = {op["id"]: op for op in operations}

    # 2. Topological sorting (DFS)
    ordered_operations = []
    visited = set()

    def visit(node_id):
        if node_id in visited:
            return

        if node_id in operations_dict:
            op = operations_dict[node_id]

            for parent in op["parents"]:
                visit(parent)

            ordered_operations.append(op)

        visited.add(node_id)

    visit(output_id)

    # 3. Forward pass
    def forward(leaf_values):

        values = dict(leaf_values)

        for op in ordered_operations:

            node_id = op["id"]
            parents = op["parents"]

            if op["op"] == "mul":
                values[node_id] = mul(
                    values[parents[0]],
                    values[parents[1]]
                )

            elif op["op"] == "add":
                values[node_id] = add(
                    values[parents[0]],
                    values[parents[1]]
                )

            elif op["op"] == "tanh":
                values[node_id] = tanh(
                    values[parents[0]]
                )

        return values

    # Initial leaf values
    leaf_values = {
        leaf["id"]: np.float64(leaf["value"])
        for leaf in leaves
        if leaf["id"] in visited
    }

    dict_values_forward = forward(leaf_values)

    # 4. Backward pass
    dict_gradients = {
        output_id: np.float64(1.0)
    }

    for op in reversed(ordered_operations):

        node_id = op["id"]
        grad = dict_gradients.get(node_id, 0.0)

        parents = op["parents"]

        if op["op"] == "tanh":

            parent = parents[0]

            local_grad = dtanh(
                dict_values_forward[node_id]
            )

            dict_gradients[parent] = (
                dict_gradients.get(parent, 0.0)
                + grad * local_grad
            )

        elif op["op"] == "mul":

            parent_a, parent_b = parents

            a = dict_values_forward[parent_a]
            b = dict_values_forward[parent_b]

            da, db = dmul(a, b)

            dict_gradients[parent_a] = (
                dict_gradients.get(parent_a, 0.0)
                + grad * da
            )

            dict_gradients[parent_b] = (
                dict_gradients.get(parent_b, 0.0)
                + grad * db
            )

        elif op["op"] == "add":

            parent_a, parent_b = parents

            da, db = dsum(
                dict_values_forward[parent_a],
                dict_values_forward[parent_b]
            )

            dict_gradients[parent_a] = (
                dict_gradients.get(parent_a, 0.0)
                + grad * da
            )

            dict_gradients[parent_b] = (
                dict_gradients.get(parent_b, 0.0)
                + grad * db
            )

    # 5. Analytical gradients
    analytical_gradients = {
        leaf["id"]: np.float64(
            dict_gradients.get(leaf["id"], 0.0)
        )
        for leaf in leaves
        if leaf["id"] in visited
    }

    # 6. Numerical gradients (forward finite differences)
    numerical_gradients = {}

    original_output = dict_values_forward[output_id]

    for leaf_id in analytical_gradients:

        perturbed_values = leaf_values.copy()
        perturbed_values[leaf_id] += h

        perturbed_output = forward(
            perturbed_values
        )[output_id]

        numerical_gradients[leaf_id] = np.float64(
            (perturbed_output - original_output) / h
        )

    # 7. Maximum absolute error
    errors = [
        abs(
            analytical_gradients[key]
            - numerical_gradients[key]
        )
        for key in analytical_gradients
    ]

    max_error = np.float64(
        max(errors, default=0.0)
    )

    return (
        analytical_gradients,
        numerical_gradients,
        max_error
    )

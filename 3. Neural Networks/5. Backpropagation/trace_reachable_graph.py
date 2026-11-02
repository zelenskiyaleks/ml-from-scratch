def trace_reachable_graph(nodes: list, output_id: str) -> tuple:
    """
    Trace all nodes in a computational graph that contribute to a given output.

    Args:
        nodes: List of dictionaries describing graph nodes.
            Each dictionary contains:
                - id: Unique node identifier.
                - parents: List of IDs of the node's parent nodes.
                - data, grad, op: Additional node information that is not
                  required for reachability analysis.
        output_id: ID of the output node from which to start tracing.

    Returns:
        Tuple containing:
            - reachable_nodes: List of reachable node IDs in their original
              order in the input nodes list.
            - edges: List of parent-to-child edges, where each edge is
              represented as [parent_id, child_id].

    Algorithm:
        1. Build a mapping between each node ID and its parent IDs.
        2. Start from the output node and traverse the graph backwards
           using a stack.
        3. Mark each visited node as reachable to avoid processing it
           more than once.
        4. Add all parent nodes of each visited node to the stack.
        5. Filter the original node list to preserve the input order
           of reachable node IDs.
        6. Build parent-to-child edges for all reachable nodes, preserving
           the original node order and the order of each node's parents.

    Notes:
        - The output node itself is included in reachable_nodes.
        - Disconnected nodes are excluded.
        - Repeated edges are preserved.
        - The traversal follows parent references backwards, but returned
          edges are oriented from parent to child.
    """

    parents_map = {
        node['id']: node['parents']
        for node in nodes
    }

    reachable = set()
    stack = [output_id]

    while stack:
        node_id = stack.pop()

        if node_id in reachable:
            continue

        reachable.add(node_id)

        for parent in parents_map[node_id]:
            stack.append(parent)

    reachable_nodes = [
        node['id']
        for node in nodes
        if node['id'] in reachable
    ]

    edges = []

    for node in nodes:
        if node['id'] in reachable:
            for parent in node['parents']:
                edges.append([parent, node['id']])

    return reachable_nodes, edges
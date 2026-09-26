def calculate_covariance_matrix(
    vectors: list[list[float]]
) -> list[list[float]]:
    """
    Calculate the covariance matrix for a set of feature vectors.

    Args:
        vectors: A list of vectors, where each vector represents
            a feature and contains its observations.

    Returns:
        The covariance matrix as a list of lists.
    """

    list_mean = [sum(vector) / len(vector) for vector in vectors]

    n_features = len(vectors)
    n_observations = len(vectors[0])

    res = [n_features * [0] for _ in range(n_features)]

    for i in range(n_features):
        for j in range(n_features):
            covariance = sum(
                (vectors[i][k] - list_mean[i]) *
                (vectors[j][k] - list_mean[j])
                for k in range(n_observations)
            ) / (n_observations - 1)

            res[i][j] = covariance
            res[j][i] = covariance

    return res
"""
Matrix-Vector Dot Product
=========================

Source: Deep-ML
Category: Linear Algebra
Difficulty: Easy

Problem
-------
Implement a function that computes the dot product of a matrix and
a vector.

A matrix (list of lists) can be multiplied by a vector (list) only
if the number of columns in the matrix equals the length of the vector.

For an n x m matrix and a vector of length m, the result is a vector
of length n.

Example
-------
Input:
    a = [[1, 2], [2, 4]]
    b = [1, 2]

Output:
    [5, 10]

Approach
--------
For each row of the matrix, calculate its dot product with the vector:

    result[i] = sum(a[i][j] * b[j])

If the dimensions are incompatible, return -1.

Complexity
----------
Time:  O(n * m)
Space: O(n)

where:
    n = number of rows in the matrix
    m = number of columns in the matrix
"""


def matrix_dot_vector(
    a: list[list[int | float]],
    b: list[int | float]
) -> list[int | float] | int:
    """Calculate the dot product of a matrix and a vector."""

    if len(a[0]) != len(b):
        return -1

    result = []

    for i in range(len(a)):
        row_result = sum(a[i][k] * b[k] for k in range(len(b)))
        result.append(row_result)

    return result
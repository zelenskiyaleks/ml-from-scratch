"""
Scalar Multiplication of a Matrix
=================================

Source: Deep-ML
Category: Linear Algebra
Difficulty: Easy

Problem
-------
Implement a function that multiplies a matrix by a scalar and returns
the resulting matrix.

Each element of the matrix is multiplied independently by the scalar.

Example
-------
Input:
    matrix = [[1, 2],
              [3, 4]]

    scalar = 2

Output:
    [[2, 4],
     [6, 8]]

Approach
--------
Iterate through every element of the matrix and multiply it by the
given scalar.

Complexity
----------
Time:  O(m * n)
Space: O(1)

where:
    m = number of rows
    n = number of columns

Note:
    The matrix is modified in place, so the returned matrix uses the
    same underlying list.
"""


def scalar_multiply(
    matrix: list[list[int | float]],
    scalar: int | float
) -> list[list[int | float]]:
    """Multiply every element of a matrix by a scalar."""

    rows = len(matrix)
    columns = len(matrix[0])

    for i in range(rows):
        for j in range(columns):
            matrix[i][j] *= scalar

    return matrix
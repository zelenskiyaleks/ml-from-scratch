"""
Transpose of a Matrix
======================

Source: Deep-ML
Category: Linear Algebra
Difficulty: Easy

Problem
-------
Implement a function that computes the transpose of a given 2D matrix.

The transpose is formed by turning the rows of a matrix into columns
and the columns into rows.

For an m x n matrix, the transpose has shape n x m.

Example
-------
Input:
    a = [[1, 2, 3],
         [4, 5, 6]]

Output:
    [[1, 4],
     [2, 5],
     [3, 6]]

Approach
--------
For each column of the original matrix, create a new row containing
the corresponding elements from all rows.

Complexity
----------
Time:  O(m * n)
Space: O(m * n)

where:
    m = number of rows
    n = number of columns
"""


def transpose_matrix(
    a: list[list[int | float]]
) -> list[list[int | float]]:
    """Return the transpose of a 2D matrix."""

    result = []

    for i in range(len(a[0])):
        row_result = []

        for j in range(len(a)):
            row_result.append(a[j][i])

        result.append(row_result)

    return result
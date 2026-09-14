"""
Reshape Matrix
==============

Source: Deep-ML
Category: Linear Algebra
Difficulty: Easy

Problem
-------
Implement a function that reshapes a given matrix into a specified shape.

The reshape operation preserves the order of elements and changes only
the dimensions of the matrix.

If the matrix cannot be reshaped into the requested shape, return an
empty list.

Example
-------
Input:
    a = [[1, 2, 3, 4],
         [5, 6, 7, 8]]

    new_shape = (4, 2)

Output:
    [[1, 2],
     [3, 4],
     [5, 6],
     [7, 8]]

Approach
--------
The matrix is traversed in row-major order.

For each element, its position in the original matrix is calculated
using integer division and modulo:

    row = index // number_of_columns
    column = index % number_of_columns

The element is then placed into the corresponding position of the
new matrix.

Before reshaping, we verify that the number of elements in the
original and target shapes is the same.

Complexity
----------
Time:  O(m * n)
Space: O(m * n)

where:
    m = number of rows in the original matrix
    n = number of columns in the original matrix
"""


def reshape_matrix(
    a: list[list[int | float]],
    new_shape: tuple[int, int]
) -> list[list[int | float]]:
    """Reshape a matrix into the specified shape."""

    old_shape = (len(a), len(a[0]))

    if old_shape[0] * old_shape[1] != new_shape[0] * new_shape[1]:
        return []

    result = []
    num_el = 0

    for i in range(new_shape[0]):
        row_result = []

        for j in range(new_shape[1]):
            row = num_el // old_shape[1]
            column = num_el % old_shape[1]

            row_result.append(a[row][column])
            num_el += 1

        result.append(row_result)

    return result
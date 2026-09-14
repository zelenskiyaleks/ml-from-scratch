"""
Calculate Mean by Row or Column
===============================

Source: Deep-ML
Category: Linear Algebra
Difficulty: Easy

Problem
-------
Implement a function that calculates the mean of a matrix either by row
or by column, based on the specified mode.

The function takes a matrix and a mode ('row' or 'column') and returns
a list containing the mean of each row or column.

Example
-------
Input:
    matrix = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]

    mode = 'column'

Output:
    [4.0, 5.0, 6.0]

Approach
--------
For row mode, iterate through each row and calculate its mean.

For column mode, swap the matrix dimensions and access the elements
using matrix[j][i] instead of matrix[i][j].

The mean is calculated by adding each element divided by the number
of elements in the corresponding row or column.

Complexity
----------
Time:  O(m * n)
Space: O(n)

where:
    m = number of rows
    n = number of columns
"""


def calculate_matrix_mean(
    matrix: list[list[float]],
    mode: str
) -> list[float]:
    """Calculate the mean of each row or column."""

    result = []

    rows = len(matrix)
    columns = len(matrix[0])

    if mode == "column":
        rows, columns = columns, rows

    for i in range(rows):
        mean = 0

        for j in range(columns):
            if mode == "row":
                mean += matrix[i][j] / columns
            else:
                mean += matrix[j][i] / columns

        result.append(mean)

    return result
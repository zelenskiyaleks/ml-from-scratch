"""
Calculate Eigenvalues of a Matrix
=================================

Source: Deep-ML
Category: Linear Algebra
Difficulty: Medium

Problem
-------
Implement a function that calculates the eigenvalues of a 2x2 matrix.

The function should return the eigenvalues sorted from highest to lowest.

Example
-------
Input:
    matrix = [[2, 1],
              [1, 2]]

Output:
    [3.0, 1.0]

Approach
--------
For a 2x2 matrix, the eigenvalues are the roots of the characteristic
equation:

    lambda^2 - trace(A) * lambda + det(A) = 0

For:

    A = [[a, b],
         [c, d]]

the trace is:

    trace(A) = a + d

and the determinant is:

    det(A) = a * d - b * c

The eigenvalues are then calculated using the quadratic formula.

Complexity
----------
Time:  O(1)
Space: O(1)

The calculation operates on a fixed-size 2x2 matrix.
"""

import math


def calculate_eigenvalues(
    matrix: list[list[float | int]]
) -> list[float]:
    """Calculate the eigenvalues of a 2x2 matrix."""

    a = 1
    b = -(matrix[0][0] + matrix[1][1])
    c = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    discriminant = math.sqrt(b**2 - 4 * a * c)

    eigenvalues = [
        (-b + discriminant) / 2,
        (-b - discriminant) / 2
    ]

    return eigenvalues
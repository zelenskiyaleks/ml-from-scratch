"""
Calculate 2x2 Matrix Inverse
============================

Source: Deep-ML
Category: Linear Algebra
Difficulty: Easy

Problem
-------
Calculate the inverse of a 2x2 matrix.

For a matrix:

    [[a, b],
     [c, d]]

the inverse exists only when the determinant is non-zero.

The inverse is calculated as:

    A⁻¹ = 1 / det(A) * [[d, -b],
                        [-c,  a]]

If the determinant equals zero, return None.

Example
-------
Input:
    [[4, 7],
     [2, 6]]

Output:
    [[0.6, -0.7],
     [-0.2, 0.4]]

Approach
--------
1. Calculate the determinant:

       det(A) = ad - bc

2. If the determinant is zero, the matrix is not invertible.

3. Swap the elements on the main diagonal and change
   the signs of the elements on the secondary diagonal.

4. Divide every element by the determinant.
"""


def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """Calculate the inverse of a 2x2 matrix."""

    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    if det == 0:
        return None

    return [
        [matrix[1][1] / det, -matrix[0][1] / det],
        [-matrix[1][0] / det, matrix[0][0] / det]
    ]
"""
Matrix Times Matrix
===================

Source: Deep-ML
Category: Linear Algebra
Difficulty: Medium

Problem
-------
Multiply two matrices A and B.

Matrix multiplication is valid only when the number of
columns in A equals the number of rows in B.

For compatible matrices:

    C = A × B

where each element C[i][j] is the dot product of
the i-th row of A and the j-th column of B.

If the matrix dimensions are incompatible, return -1.

Example
-------
Input:
    A = [[1, 2],
         [2, 4]]

    B = [[2, 1],
         [3, 4]]

Output:
    [[8, 9],
     [16, 18]]

Approach
--------
1. Check that the inner dimensions match:
       columns(A) == rows(B)

2. Create the resulting matrix with shape:
       rows(A) × columns(B)

3. Calculate every element using:
       C[i][j] = sum(A[i][k] * B[k][j])
"""


def matrixmul(
    a: list[list[int | float]],
    b: list[list[int | float]]
) -> list[list[int | float]] | int:
    """Multiply two matrices or return -1 for incompatible shapes."""

    if len(a[0]) != len(b):
        return -1

    l_a_1 = len(a)
    l_a_2 = len(a[0])
    l_b_2 = len(b[0])

    res = list()

    for i in range(l_a_1):
        res_l = list()

        for j in range(l_b_2):
            res_i = 0

            for k in range(l_a_2):
                res_i += a[i][k] * b[k][j]

            res_l.append(res_i)

        res.append(res_l)

    return res
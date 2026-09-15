"""
Matrix Transformation
======================

Source: Deep-ML
Category: Linear Algebra
Difficulty: Medium

Problem
-------
Transform a matrix A using the operation:

    T⁻¹AS

where T and S are invertible matrices.

If either T or S is not invertible, return -1.

Example
-------
Input:
    A = [[1, 2],
         [3, 4]]

    T = [[2, 0],
         [0, 2]]

    S = [[1, 1],
         [0, 1]]

Output:
    [[0.5, 1.5],
     [1.5, 3.5]]

Approach
--------
NumPy is used for matrix inversion and matrix multiplication.

First, check that both T and S are invertible. Then calculate:

    T⁻¹AS

The resulting NumPy array is converted back to a Python list.

Complexity
----------
Depends on the matrix size n.

Matrix inversion: O(n³)
Matrix multiplication: O(n³)
"""


import numpy as np


def transform_matrix(A, T, S):
    """Transform matrix A using T⁻¹AS."""

    try:
        T_inv = np.linalg.inv(T)
        np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return -1

    return (T_inv @ np.array(A) @ np.array(S)).tolist()
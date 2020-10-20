import numpy as np

def solve(matrix):
    """Takes an extended system matrix."""

    A = matrix.copy()
    for i in range(A.shape[0]):
        swap_cur_row_with = i
        for k in range(i, A.shape[0]):
            if A[k][i] != 0:
                swap_cur_row_with = k
        if swap_cur_row_with - i != 0:
            A[i], A[swap_cur_row_with] = A[swap_cur_row_with], A[i].copy()
        A[i] = A[i] / A[i][i]
        for j in range(i + 1, A.shape[0]):
            A[j] = A[j] - A[i]*A[j][i]

    answ = np.zeros(A.shape[0])
    for i in range(A.shape[0]-1, -1, -1):
        xi = A[i][A.shape[0]]
        for j in range(i+1, A.shape[0]):
            xi -= A[i][j]*answ[j]
        answ[i] = xi

    return answ

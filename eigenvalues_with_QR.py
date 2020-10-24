import numpy as np

def get_eigenvalues(A):
    temp = A
    while(True):
        Q, R = np.linalg.qr(temp)
        temp = np.dot(R, Q)
        LU = temp - np.diag(np.diag(temp))
        if((abs(LU) <= 1e-10).all()):
            break
    return np.diag(temp)
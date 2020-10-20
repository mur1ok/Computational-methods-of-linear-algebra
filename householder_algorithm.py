import numpy as np

def solve(matrix, f):
    A = matrix.copy()
    b = f.copy()
    
    n = A.shape[0]
    for j in range(1, n + 1):
        H = np.eye(n)   
        a = A[j - 1:, j - 1].copy()
        r = np.linalg.norm(a)
        w = np.array([])
        if r == 0 or np.array_equal(a[a==0], a[1:]):
            w = np.array([0])
        elif r != 0 and a[0] == 0:
            a[0] -=  r
            w = a/np.linalg.norm(a)
        else:
            a[0] += r*np.sign(a[0])
            w = a/np.linalg.norm(a)
        H[j - 1:, j - 1:] -= 2*np.outer(w, w)
        A = np.dot(H, A)
        b = np.dot(H, b)

    answ = np.array([0.0 for i in range(n)])
    for i in range(n-1, -1, -1):
        xi = b[i]
        for j in range(i+1, n):
            xi -= A[i][j]*answ[j]
        answ[i] = xi/A[i][i]
    return answ
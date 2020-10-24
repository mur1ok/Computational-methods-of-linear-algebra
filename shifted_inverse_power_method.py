import numpy as np

def get_eigenvalue(A, x, alpha):
    n = A.shape[0]
    y = np.dot(np.linalg.inv(A - alpha*np.eye(n)), x)
    c_old = np.dot(y, x)/np.dot(x, x)
    converge = False

    while(not converge):
        x = y/np.linalg.norm(y)
        y = np.dot(np.linalg.inv(A - alpha*np.eye(n)), x)
        c_new = np.dot(y, x)/np.dot(x, x)
        converge = np.linalg.norm(c_new - c_old) <= 1e-13
        c_old = c_new

    return 1/c_old + alpha
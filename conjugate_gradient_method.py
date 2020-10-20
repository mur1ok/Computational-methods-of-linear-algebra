import numpy as np

def solve(A, b, x):
    r = b - np.dot(A, x)
    s = r
    r_old = np.dot(r, r)
    for _ in range(len(b)):
        g = np.dot(A, s)
        alpha = r_old / np.dot(s, g)
        x = x + np.dot(alpha, s)
        r = r - np.dot(alpha, g)
        r_new = np.dot(r, r)
        s = r + (r_new/r_old)*s
        r_old = r_new
    return x
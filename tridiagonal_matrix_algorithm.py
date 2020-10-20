import numpy as np

def solve(matrix):
    """
    Takes an extended system matrix.

    b - main diagonal;
    a - under the main diagonal;
    c - over the main diagonal.

    """

    a = [0]
    b = []
    c = []
    f = []
    n = matrix.shape[0]

    for i in range(n):
        b.append(matrix[i][i])
        f.append(matrix[i][n])
        if i != 0:
            a.append(matrix[i][i-1])
        if i != n-1:
            c.append(matrix[i][i + 1])
    c.append(0)

    alpha = [-c[0]/b[0]]
    beta = [f[0]/b[0]]

    answ = np.zeros(n)

    for i in range(1,n-1):
        alpha.append(-c[i]/(a[i]*alpha[i - 1] + b[i]))
        beta.append((f[i] - a[i]*beta[i - 1])/(a[i]*alpha[i - 1] + b[i]))
    beta.append((f[n-1] - a[n-1]*beta[n-2])/(a[n-1]*alpha[n-2] + b[n-1]))
    
    answ[n-1] = beta[n-1]

    for i in range(n-2, -1, -1):
        answ[i] = alpha[i]*answ[i+1] + beta[i]

    return answ

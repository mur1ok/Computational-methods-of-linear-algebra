import numpy as np

def solve(A, b, x_0, alpha, beta, eps = 1e-9):
    x_1 = x_0 - (2/(beta + alpha))*(np.dot(A, x_0) - b)
    omega_1 = -(beta - alpha)/(beta + alpha)
    omega_old = omega_1
    converge = False
    
    while(not converge):
        omega_new = 1/(2*(1/omega_1) - omega_old)
        x_2 = x_1 + omega_old*omega_new*(x_1 - x_0) - (2/(beta + alpha))*(
                1 + omega_old*omega_new)*(np.dot(A, x_1) - b)
        omega_old = omega_new
        x_0, x_1 = x_1, x_2
        converge = np.linalg.norm(x_1 - x_0) <= eps

    return x_1
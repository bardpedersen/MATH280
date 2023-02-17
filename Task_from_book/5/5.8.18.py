import numpy as np


def inverse_pow_method(p1, v0, n1, alpha):
    """
    inverse_pow_method(P, v0, N):
        Input:
            P:   transition matrix
            v0:  initial vector of probabilities
            N:   number of iterations
            alpa: estimation of the eigenvalue
        Output:
            v_k: approximation of the eigenvalue
            my:  approaches the dominant eigenvalue
    """
    v = v0
    v_k = 0
    my = 0
    for i in range(n1):
        i = np.shape(p1)
        inverse = np.linalg.inv(p1-alpha*np.identity(i[1]))
        y_k = np.matmul(inverse, v)
        my = np.max(y_k)
        v_k = alpha + 1 / my
        v = (1 / my) * y_k
    return v_k, my


P = np.array([[8, 0, 12], [1, -2, 1], [0, 3, 0]])
X0 = np.array([[1], [0], [0]])
alpa = -1.4
k1 = 1
k2 = 2
k3 = 3

pm1, my0 = inverse_pow_method(P, X0, k1, alpa)
pm2, my1 = inverse_pow_method(P, X0, k2, alpa)
pm3, my2 = inverse_pow_method(P, X0, k3, alpa)

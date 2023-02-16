import numpy as np


def pow_method(p1, v0, n1):
    """
    pow_method(P, v0, N):
        Input:
            P:   transition matrix
            v0:  initial vector of probabilities
            N:   number of iterations
        Output:
            v:   final vector of probabilities v=P^n * v0
    """
    v = v0
    for i in range(n1):
        v = np.matmul(p1, v)
        my = np.max(v)
        v = 1 / my * v

    return v, my,




P = np.array([[5, 2], [2, 2]])
X0 = np.array([[1], [0]])
k1 = 1
k2 = 2
k3 = 3
k4 = 4
k5 = 5
k6 = 6
k7 = 7
pm1, my0 = pow_method(P, X0, k1)
pm2, my1 = pow_method(P, X0, k2)
pm3, my2 = pow_method(P, X0, k3)
pm4, my3 = pow_method(P, X0, k4)
pm5, my4 = pow_method(P, X0, k5)
v_T = np.transpose(pm5)
temp = np.matmul(P, pm5)
R = (np.matmul(v_T*temp))/(np.matmul(v_T*pm5))

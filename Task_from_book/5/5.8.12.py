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
            my:  approaches the dominant eigenvalue
    """
    v = v0
    my = 0
    for i in range(n1):
        v = np.matmul(p1, v)
        my = np.max(v)
        v = 1 / my * v

    return v, my


def rayleigh_quotient(a, v_f):
    """
    rayleigh_quotient(a, v_f)
        Input:
            a:   matrix
            v_f:  initial vector of probabilities
        Output:
            r:   approximation to eigenvalue
    """
    v_t = np.transpose(v_f)
    temp = np.matmul(a, v_f)
    r = (np.matmul(v_t, temp))/(np.matmul(v_t, v_f))

    return r


P = np.array([[-3, 2], [2, 0]])
X0 = np.array([[1], [0]])
k1 = 1
k2 = 2
k3 = 3
k4 = 4
k5 = 5
pm1, my0 = pow_method(P, X0, k1)
pm2, my1 = pow_method(P, X0, k2)
pm3, my2 = pow_method(P, X0, k3)
pm4, my3 = pow_method(P, X0, k4)
pm5, my4 = pow_method(P, X0, k5)

r1 = rayleigh_quotient(P, pm1)
r2 = rayleigh_quotient(P, pm2)
r3 = rayleigh_quotient(P, pm3)
r4 = rayleigh_quotient(P, pm4)
r5 = rayleigh_quotient(P, pm5)

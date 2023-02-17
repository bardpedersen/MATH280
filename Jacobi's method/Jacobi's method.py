import numpy as np


def jacobi(a, b, tol):
    """
    jacobi(a, b, tol):
        Input:
            a:  matrix
            b:  vector
            tol: tolerance
        Output:
            x:   final answers to solved matrix*v=v with x, y and z
            it:  number of iteration
    """
    n = b.shape[0]
    x = np.zeros(n)
    max_it = 100
    d = np.diagonal(a)
    n1 = a - np.diag(d)

    rem = np.linalg.norm(b - a @ x)

    it = 0
    while rem > tol and it < max_it:
        x = (b - n1 @ x) / d
        rem = np.linalg.norm(b - a @ x)
        it = it + 1

    return x, it


A = np.array([[6, -2,  3],
             [-3,  9,  1],
             [2, -1, -7]])
B = np.array([-1, 2, 3])

c, iteration = jacobi(A, B, 1e-10)

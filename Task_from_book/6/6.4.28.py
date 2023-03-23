import numpy as np


def vector_product_element(x_n, v_n_1):
    temp = np.dot(x_n, v_n_1) / np.dot(v_n_1, v_n_1)
    element = np.dot(temp, v_n_1)

    return element


def gram_schmidt_proces(matrix):
    ans = []
    for x_n in matrix:
        if x_n == matrix[0]:
            temp = np.array(x_n)
        else:
            temp = x_n
            for j in ans:
                ele = vector_product_element(x_n, j)
                temp -= ele
        ans.append(temp.tolist())

    return ans


if __name__ == '__main__':
    A = [[-10, 13, 7, -11], [2, 1, -5, 3], [-6, 3, 13, -3], [16, -16, -2, 5], [2, 1, -5, -7]]
    A = np.transpose(A).tolist()
    gram = gram_schmidt_proces(A)
    print(gram)

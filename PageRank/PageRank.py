import numpy as np


# %% Task 1

A = np.loadtxt('adjacency_example.txt', dtype=float)
print(A)
print('------------------')


def transition_matrix(a):
    """
    transition_matrix(A):
        Input:
            A: Adjacency matrix
        Output:
            p: Transition matrix
    """
    p = a
    for i, list1 in enumerate(p):
        sum_list = sum(list1)
        for j in range(len(list1)):
            new_number = list1[j] / sum_list
            if list1[j] != 0:
                list1[j] = new_number

    p_ans = p.transpose()
    return p_ans


P = transition_matrix(A)
q = np.allclose(np.sum(P, axis=0), 1)  # Will return True if P is a stochastic matrix


# %%  Task 2

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
    p_matrix = p1
    for i in range(n1):
        p_matrix = np.matmul(p_matrix, p1)

    v = p_matrix.dot(v0)
    return v


n = P.shape[0]  # The number of nodes
x0 = np.ones(n)/n  # initial probabilities: 1/n for every node
N = 1000  # number of iterations

print(pow_method(P, x0, N))
print('------------------')


# %% Task 3

A_realtek = np.loadtxt('adjacency_realtek.txt', dtype=float)
P_realtek = transition_matrix(A_realtek)
n_realtek = P_realtek.shape[0]  # The number of nodes
x0_realtek = np.ones(n_realtek)/n_realtek  # initial probabilities: 1/n for every node
N_realtek = 1000  # number of iterations
q_realtek = pow_method(P_realtek, x0_realtek, N_realtek)


web_name = np.loadtxt('keyvalues.txt', dtype=str)
dict1 = dict(zip(web_name[:, 1], q_realtek))
sort_dict = dict(sorted(dict1.items(), key=lambda item: item[1]))
for x in list(reversed(list(sort_dict)))[0:5]:
    print(x)
print('------------------')

# %% Task 4 a)


def pow_method2(p1, v0, tol):
    """
    pow_method(P, v0, N):
        Input:
            P:   transition matrix
            v0:  initial vector of probabilities
            tol:   tolerance in error for Xk - X(k-1)
        Output:
            v:   final vector of probabilities v=P^n * v0
            i:   number of iteration to satisfy tolerance
    """
    p_matrix = p1
    x_ = v0
    run = True
    i = 0
    while run:
        i += 1
        p_matrix = np.matmul(p_matrix, p1)
        x1 = p_matrix.dot(x_)
        ans = abs(x1 - x_)
        if ans.all() <= tol:
            run = False
        x_ = x1

    v = p_matrix.dot(v0)
    return v, i


tolerance = 0.000001
q_realtek2, number_of_N = pow_method2(P_realtek, x0_realtek, tolerance)
print('it tok matrix to the power of: ', number_of_N, 'to get the right number')
dict1 = dict(zip(web_name[:, 1], q_realtek2))
sort_dict = dict(sorted(dict1.items(), key=lambda item: item[1]))
for x in list(reversed(list(sort_dict)))[0:5]:
    print(x)
print('------------------')

# %% Task 4 b)


def transition_matrix2(a):
    """
    transition_matrix(A):
        Input:
            A: Adjacency matrix
        Output:
            p: Transition matrix
    """
    p = a
    for i, list1 in enumerate(p):
        sum_list = sum(list1)
        if sum_list == 0:
            print('----------------')

        for j in range(len(list1)):
            new_number = list1[j] / sum_list
            if list1[j] != 0:
                list1[j] = new_number

    for i, list2 in enumerate(p):
        done = True
        for j in range(len(list2)):
            if list2[j] == 1 and j == i and done:
                for z in range(len(list2)):
                    list2[z] = 1/len(list2)
                done = False

    p_ans = p.transpose()
    return p_ans


A2 = np.loadtxt('adjacency_example_adjustment1.txt', dtype=float)
P2 = transition_matrix2(A2)


def adjustment2(a, prob):
    """
    adjustment2(a, prob):
        Input:
            a: adjacency matrix
            prob: weight in calculation for adjustment2
        Output:
            g: google matrix
    """
    size = np.shape(a)
    m = [[1/size[0]] * size[1]] * size[0]
    m = np.array(m)
    matrix1 = prob * a
    matrix2 = (1-prob) * m
    g = np.add(matrix1, matrix2)
    return g


Probability = 0.85
P_realtek2 = transition_matrix2(A_realtek)
answer_adjustment2 = adjustment2(P_realtek2, Probability)
q_realtek21, number_of_N2 = pow_method2(P_realtek2, x0_realtek, tolerance)
print('it tok matrix to the power of: ', number_of_N2, 'to get the right number')
dict1 = dict(zip(web_name[:, 1], q_realtek21))
sort_dict = dict(sorted(dict1.items(), key=lambda item: item[1]))
for x in list(reversed(list(sort_dict)))[0:5]:
    print(x)
print('------------------')

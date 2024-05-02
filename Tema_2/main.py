import copy
import math
import random

import numpy as np


def set_machine_precision():
    t = 4
    while t < 5:
        t = int(input("Introduceti t>=5 pentru care 10^(-t) este precizia masinii: "))
    return t

def random_n():
    n = random.randint(1, 10)
    print(f'Numarul aleator este {n}, matricea generata va avea aceasta dimensiunea')
    return n
def randomize_A_matrix(n):
    A = [[0.0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = random.randint(0, 10)
    return A

def randomize_b_vector(n):
    b = [0.0] * n
    for i in range(n):
        b[i] = random.randint(0, 10)
    return b

def matrix_determinant(A, n):
    det = 1.0
    for i in range(n):
            if A[i][i] == 0:
                return 0
            det *= A[i][i]
    return det

def print_det(A, A_init, n):
    print("--Determinant--")
    check = matrix_determinant(A, n) == np.linalg.det(A_init)
    difference = abs(matrix_determinant(A, n) - np.linalg.det(A_init))
    print(f"{matrix_determinant(A, n)} == {np.linalg.det(A_init)} is {check}")
    print(f"Diferenta este {difference}")
def verify_division_by_epsilon(number, epsilon):
    if abs(number) <= epsilon:
        print(f"Nu se poate imparti, număr = {number} mai mic decât epsilon = {epsilon}!")
        return False
    return True

def crout_decomposition(A):
    n = len(A)
    for j in range(n):

        for i in range(j, n):
            for k in range(j):
                A[i][j] -= A[i][k] * A[k][j]
        if verify_division_by_epsilon(A[j][j], 10**(-t)) == False:
            return False
        for i in range(j + 1, n):
            for k in range(j):
                A[j][i] -= A[j][k] * A[k][i]
            A[j][i] /= A[j][j]
    return A

def lower_substitution(A, b,t):
    n = len(A)
    y = [0.0] * n

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= A[i][j] * y[j]
        if verify_division_by_epsilon(A[i][i], 10**(-t)) == False:
            return False
        y[i] /= A[i][i]
    #formula 3
    return y

def upper_substitution(A, y):
    n = len(A)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
    return x

def calculate_norm_manual(A_init, x, b):
    n = len(A_init)
    result = [0.0] * n

    for i in range(n):
        for j in range(n):
            result[i] += A_init[i][j] * x[j]
    residual = [result[i] - b[i] for i in range(len(b))]
    residual_norm_squared = sum([residual[i] ** 2 for i in range(len(residual))])
    return math.sqrt(residual_norm_squared)


def print_norm(A_init,x, b):
    print("--Norma--")
    print(f" Manual {calculate_norm_manual(A_init, x, b)}")
    print(f" NumPy {np.linalg.norm(np.dot(A_init, x) - b)}")

def solve_for_random():
    n = random_n()
    b = randomize_b_vector(n)
    A = randomize_A_matrix(n)
    A_init = copy.deepcopy(A)
    A = crout_decomposition(A)
    if A  == False:
        return
    y = lower_substitution(A, b, t)
    x = upper_substitution(A, y)

    print("--Solutia--")
    print(x)
    print_det(A, A_init, n)
    print_norm(A_init, x, b)
    solve_using_numpy(A_init, b, x)

def solve_for_given():
    b = [2, 2, 2]
    A = [[2.5, 2, 2], [5, 6, 5], [5, 6, 6.5]]
    A_init = copy.deepcopy(A)
    A = crout_decomposition(A)
    y = lower_substitution(A, b,t)
    x = upper_substitution(A, y)
    print("--A-init--")
    print(A_init)
    print("--L U--")
    print(A)
    print("--Solutia--")
    print(x)
    print_det(A, A_init, 3)
    print_norm(A_init, x, b)
    solve_using_numpy(A_init, b, x)

def solve_using_numpy(A_init, b_init, x_LU):
    x_lib = np.linalg.solve(A_init, b_init)
    norm_residual_x = np.linalg.norm(x_LU - x_lib)
    norm_residual_A = np.linalg.norm(x_LU - np.dot(np.linalg.inv(A_init), b_init))
    print("\n-----------------------Utilizand Numpy-----------------------------")
    print("--Solutia--")
    print(x_lib)
    print("--A--")
    print(A_init)
    print("--Inversa--")
    print(np.linalg.inv(A_init))
    print("--Determinant--")
    print(np.linalg.det(A_init))
    print("--Norma 1--")
    print(norm_residual_x)
    print("--Norma 2--")
    print(norm_residual_A)

if __name__ == '__main__':
    t = set_machine_precision()
    print("-----------------------Exemplu random-----------------------------")
    solve_for_random()


    print("\n-----------------------Exemplu din lab-----------------------------")
    solve_for_given()


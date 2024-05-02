import copy
import math
import random

import numpy as np


def set_machine_precision():
    t = 4
    while t < 5:
        t = int(input("Introduceti t>=5 pentru care 10^(-t) este precizia masinii: "))
    return t

def verify_division_by_epsilon(number, epsilon):
    if abs(number) <= epsilon:
        print(f"Nu se poate imparti, număr = {number} mai mic decât epsilon = {epsilon}!")
        return False
    return True

def verify_initial_matrix(A,epsilon):
    n = len(A)
    for i in range(n):
        if verify_division_by_epsilon(A[i][i],epsilon) == False:
            return False
    return True

def random_n():
    n = random.randint(2, 10)
    print(f'Numarul aleator este {n}, matricea generata va avea aceasta dimensiunea')
    return n
def randomize_A_matrix(n):
    A = [[0.0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = random.randint(-100, 100)
    return A

def randomize_s_vector(n):
    s = [0.0] * n
    for i in range(n):
        s[i] = random.randint(-100, 100)
    return s
def create_identity_matrix(n):
    I = [[float(i == j) for i in range(n)] for j in range(n)]
    return I
def calculate_b(A,s):
    n = len(A)
    b = [0] * n
    for i in range(0, n):
        temp = 0
        for j in range(0, n):
            temp += s[j] * A[i][j]
        b[i] = temp
    return b

def calculate_sigma(A,r,n):
    sum = 0.0
    for j in range(r,n):
        sum += A[j][r]**2
    return sum

def A_x_B(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            element = 0
            for k in range(len(B)):
                element += A[i][k] * B[k][j]
            row.append(element)
        result.append(row)

    return result

def A_minus_B(A,B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] - B[i][j])
        result.append(row)
    return result

def QR_decomposition(A,b,epsilon):
    n = len(A)
    Q = create_identity_matrix(n)

    for r in range(n - 1):
        I = create_identity_matrix(n)

        #Calculate sigma,beta,k
        sigma = calculate_sigma(A,r,n)

        sign = -1 if A[r][r]<0 else 1
        k = -sign * math.sqrt(sigma)
        beta = sigma - k * A[r][r]

        if verify_division_by_epsilon(beta, epsilon) == False:
            return False, False, False

        #Calculate vector u, u_t
        u = [0.0] * n
        u[r] = A[r][r] - k
        for i in range (r+1,n):
            u[i] = A[i][r]

        #Calculate matrix v/beta
        V = create_identity_matrix(n)
        for i in range(n):
            for j in range(n):
                if i<=r-1:
                    V[i][j] = 0
                elif i>=r and j<=r-1:
                    V[i][j] = 0
                else:
                    V[i][j] = u[i] * u[j] / beta

        print(f"v: {V}")

        #calculate new b
        gama = 0.0
        sum = 0.0
        for i in range(r, n):
            sum += u[i] * b[i]
        gama = sum / beta
        for i in range(r, n):
            b[i] = b[i] - gama * u[i]

        #Calculate Pr
        Pr = A_minus_B(I,V)

        #Calculate new  A,b,Q
        A = A_x_B(Pr,A)
        Q = A_x_B(Q,Pr)

    return  A,b,Q

def qr_decomposition_using_numpy(matrix):
    Q, R = np.linalg.qr(matrix)
    return Q, R

def solve_upper_triangular_system(A, b):
    n = len(A)
    x = [0] * n

    for i in range(n - 1, -1, -1):
        sum = 0
        for j in range(i + 1, n):
            sum += A[i][j] * x[j]
        x[i] = (b[i] - sum) / A[i][i]

    return x

def calculate_error(x_qr,x_h):
    n = len(x_qr)
    error = 0
    for i in range(n):
        error += (x_qr[i] - x_h[i]) ** 2
    return math.sqrt(error)

def calculate_error_1_2(A_init, x, b_init):
    Ax_minus_b_init = []
    for i in range(len(A_init)):
        row_sum = sum(A_init[i][j] * x[j] for j in range(len(x))) - b_init[i]
        Ax_minus_b_init.append(row_sum)
    error = sum(row_sum ** 2 for row_sum in Ax_minus_b_init)
    return math.sqrt(error)

def calculate_error_3_4(x, s):
    x_minus_s = [x[i] - s[i] for i in range(len(x))]
    error = sum(val ** 2 for val in x_minus_s) ** 0.5 / sum(val ** 2 for val in s)
    return math.sqrt(error)
def calculate_errors(A_init, b_init, x_qr, x_h, s):

    print("Calculating errors using manual calculations:")

    print(f"|x_numpy - x_qr|_2: {calculate_error(x_qr, x_h)}")

    error_1 = calculate_error_1_2(A_init, x_h, b_init)
    print(f"|A_init * x_h - b_init| = {error_1} <= {10 ** (-6)} == {error_1 <= 10 ** (-6)}")

    error_2 = calculate_error_1_2(A_init, x_qr, b_init)
    print(f"|A_init * x_qr - b_init| = {error_2} <= {10 ** (-6)} == {error_2 <= 10 ** (-6)}")

    error_3 = calculate_error_3_4(x_h, s)
    print(f"|x_h-s| / |s| = {error_3} <= {10 ** (-6)} == {error_3 <= 10 ** (-6)}")

    error_4 = calculate_error_3_4(x_qr, s)
    print(f"|x_qr-s| / |s| = {error_4} <= {10 ** (-6)} == {error_4 <= 10 ** (-6)}")

    print("\nCalculating errors using numpy:")

    print(f"|x_numpy - x_qr|_2: {np.linalg.norm(x_h - x_qr)}")

    error_1 = np.linalg.norm(np.dot(A_init, x_h) - b_init)
    print(f"|A_init * x_h - b_init| = {error_1} <= {10**(-6)} == {error_1 <= 10**(-6)}")

    error_2 = np.linalg.norm(np.dot(A_init, x_qr) - b_init)
    print(f"|A_init * x_qr - b_init| = {error_2} <= {10**(-6)} == {error_2 <= 10**(-6)}")

    error_3 = np.linalg.norm(x_h - s) / np.linalg.norm(s)
    print(f"|x_h-s| / |s| = {error_3} <= {10**(-6)} == {error_3 <= 10**(-6)}")

    error_4 = np.linalg.norm(x_qr - s) / np.linalg.norm(s)
    print(f"|x_qr-s| / |s| = {error_4} <= {10**(-6)} == {error_4 <= 10**(-6)}")


def calculate_inverse(A,Q):
    n = len(A)
    A_inv = create_identity_matrix(n)
    for i in range(n):
        y = solve_upper_triangular_system(A, Q[i])
        for j in range(n):
            A_inv[j][i] = y[j]
    return A_inv

def matrix_norm_difference(A, B):
    norm = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            norm += (A[i][j] - B[i][j]) ** 2
    norm = math.sqrt(norm)
    return norm



if __name__ == '__main__':
    epsilon = 10 ** (-set_machine_precision())
    n = random_n()
    s = randomize_s_vector(n)
    A = randomize_A_matrix(n)
    A_init = copy.deepcopy(A)

    #0 Verify initial matrix
    if verify_initial_matrix(A,epsilon) == False:
        print("Matricea are elemente |a_i_i| <= epsilon")
        exit()

    #1. Calculate b
    b_init = calculate_b(A,s)
    print("b_init:")
    print(b_init)

    b= copy.deepcopy(b_init)
    #2. QR decomposition
    A,b,Q = QR_decomposition(A,b,epsilon)
    if A == False:
        print("Nu se poate imparti, alegem altă matrice")
        exit()

    print("----------------Manual----------------")
    print("A:")
    for i in range(n):
        print(A[i])
    print("b:")
    print(b)
    print("Q:")
    for i in range(n):
        print(Q[i])
    verify_initial_matrix(A,epsilon)
    #3. Solve upper triangular system
    x_qr = solve_upper_triangular_system(A, b)
    print("x:")
    print(x_qr)


    print ("----------------Numpy----------------")
    Q_numpy, R_numpy = np.linalg.qr(A_init)
    Qb = np.matmul(Q_numpy.T, b_init)
    x_h = np.linalg.solve(R_numpy, Qb)
    print("Q using numpy:")
    print(Q_numpy)
    print("R using numpy:")
    print(R_numpy)
    print("Qb using numpy:")
    print(Qb)
    print("x using numpy:")
    print(x_h)

    print("----------------Error----------------")
    calculate_errors(A_init,b_init,np.array(x_qr),np.array(x_h),np.array(s))

    print("----------------Inversa----------------")
    Q_transpose = np.transpose(Q)
    A_inv = calculate_inverse(A, Q)
    print("A_inv:")
    print(A_inv)
    A_inv_numpy = np.linalg.inv(A_init)
    print("A_inv_numpy:")
    print(A_inv_numpy)
    print("Verificare:")
    print(np.dot(A_init, A_inv))
    print("Norma manuala: |A_qr - A_numpy| = ", matrix_norm_difference(A_inv, A_inv_numpy))
    print("Norma numpy: ", np.linalg.norm(A_inv - A_inv_numpy))





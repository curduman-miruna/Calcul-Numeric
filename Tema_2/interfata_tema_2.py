import tkinter as tk
from tkinter import messagebox
import copy
import math
import random
import numpy as np


def set_machine_precision():
    t = int(t_entry.get())
    if t < 5:
        messagebox.showerror("Error", "Introduceți un număr valid pentru t (t >= 5)!")
        return
    return t


def random_n():
    n = random.randint(1, 150)
    result_text.insert(tk.END, f'Numarul aleator este {n}, matricea generata va avea aceasta dimensiunea\n')
    return n


def randomize_A_matrix(n):
    A = [[0.0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = random.randint(-100, 100)
    return A


def randomize_b_vector(n):
    b = [0.0] * n
    for i in range(n):
        b[i] = random.randint(-100, 100)
    return b


def matrix_determinant(A, n):
    det = 1.0
    for i in range(n):
        if A[i][i] == 0:
            return 0
        det *= A[i][i]
    return det


def print_det(A, A_init, n):
    result_text.insert(tk.END, "--Determinant--\n")
    check = matrix_determinant(A, n) == np.linalg.det(A_init)
    difference = abs(matrix_determinant(A, n) - np.linalg.det(A_init))
    result_text.insert(tk.END, f"{matrix_determinant(A, n)} == {np.linalg.det(A_init)} is {check}\n")
    result_text.insert(tk.END, f"Diferenta este {difference}\n")


def verify_division_by_epsilon(number, epsilon):
    if abs(number) <= epsilon:
        result_text.insert(tk.END, f"Nu se poate imparti, număr = {number} mai mic decât epsilon = {epsilon}!\n")
        return False
    return True


def crout_decomposition(A, t):  # Adăugăm t ca argument al funcției
    n = len(A)
    for j in range(n):
        if verify_division_by_epsilon(A[j][j], 10 ** (-t)) == False:  # Utilizăm t-ul furnizat
            return False
        for i in range(j, n):
            for k in range(j):
                A[i][j] -= A[i][k] * A[k][j]

        for i in range(j + 1, n):
            for k in range(j):
                A[j][i] -= A[j][k] * A[k][i]
            A[j][i] /= A[j][j]

    return A

def lower_substitution(A, b):
    n = len(A)
    y = [0.0] * n

    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= A[i][j] * y[j]
        y[i] /= A[i][i]
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


def print_norm(A_init, x, b):
    result_text.insert(tk.END, "--Norma--\n")
    result_text.insert(tk.END, f" Manual {calculate_norm_manual(A_init, x, b)}\n")
    result_text.insert(tk.END, f" NumPy {np.linalg.norm(np.dot(A_init, x) - b)}\n")


def solve_for_random():
    result_text.delete('1.0', tk.END)  # Clear previous results
    t = set_machine_precision()
    if t is None:
        return
    n = random_n()
    b = randomize_b_vector(n)
    A = randomize_A_matrix(n)
    A_init = copy.deepcopy(A)
    if crout_decomposition(A, t) == False:
        return
    A = crout_decomposition(A, t)
    y = lower_substitution(A, b)
    x = upper_substitution(A, y)

    result_text.insert(tk.END, "--Solutia--\n")
    result_text.insert(tk.END, f"{x}\n")
    print_det(A, A_init, n)
    print_norm(A_init, x, b)
    solve_using_numpy(A_init, b, x)


def solve_for_given():
    result_text.delete('1.0', tk.END)
    t = set_machine_precision()
    if t is None:
        return
    b = [2, 2, 2]
    A = [[2.5, 2, 2], [5, 6, 5], [5, 6, 6.5]]
    A_init = copy.deepcopy(A)
    A = crout_decomposition(A, t)
    y = lower_substitution(A, b)
    x = upper_substitution(A, y)
    result_text.insert(tk.END, "--A--\n")
    result_text.insert(tk.END, f"{A_init}\n")
    result_text.insert(tk.END, "--L U--\n")
    result_text.insert(tk.END, f"{A}\n")
    result_text.insert(tk.END, "--Solutia--\n")
    result_text.insert(tk.END, f"{x}\n")
    print_det(A, A_init, 3)
    print_norm(A_init, x, b)
    solve_using_numpy(A_init, b, x)



def solve_using_numpy(A_init, b_init, x_LU):
    x_lib = np.linalg.solve(A_init, b_init)
    norm_residual_x = np.linalg.norm(x_LU - x_lib)
    norm_residual_A = np.linalg.norm(x_LU - np.dot(np.linalg.inv(A_init), b_init))
    result_text.insert(tk.END, "\n-----------------------Utilizand Numpy-----------------------------\n")
    result_text.insert(tk.END, "--Solutia--\n")
    result_text.insert(tk.END, f"{x_lib}\n")
    result_text.insert(tk.END, "--A--\n")
    result_text.insert(tk.END, f"{A_init}\n")
    result_text.insert(tk.END, "--Inversa--\n")
    result_text.insert(tk.END, f"{np.linalg.inv(A_init)}\n")
    result_text.insert(tk.END, "--Determinant--\n")
    result_text.insert(tk.END, f"{np.linalg.det(A_init)}\n")
    result_text.insert(tk.END, "--Norma 1--\n")
    result_text.insert(tk.END, f"{norm_residual_x}\n")
    result_text.insert(tk.END, "--Norma 2--\n")
    result_text.insert(tk.END, f"{norm_residual_A}\n")


root = tk.Tk()
root.title("Calculator LU")

t_label = tk.Label(root, text="Introduceți t (t >= 5):")
t_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
t_entry = tk.Entry(root)
t_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

random_button = tk.Button(root, text="Exemplu random", command=solve_for_random)
random_button.grid(row=1, column=0, padx=5, pady=5)

given_button = tk.Button(root, text="Exemplu dat", command=solve_for_given)
given_button.grid(row=1, column=1, padx=5, pady=5)

result_text = tk.Text(root, width=50, height=20)
result_text.grid(row=2, columnspan=2, padx=5, pady=5)

root.mainloop()


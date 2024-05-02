import copy
import math
import random
import time
import traceback

import numpy as np

epsilon = 0

def set_machine_precision():
    t = int(input("Introduceti pentru care 10^(-t) este precizia masinii: "))
    return t

def verify_epsilon(number):
    if abs(number) <= epsilon:
        print(f"Nu se poate imparti, număr = {number} mai mic decât epsilon = {epsilon}!")
        return False
    return True

def read_A_from_file(file_name, condition= True):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        n = int(lines[0].strip())
        sparse_matrix = []
        check_diagonal = [0] * n

        for line in lines[1:]:
            val, row, col = map(float, line.strip().split(", "))
            row,col = int(row),int(col)
            if row == col:
                check_diagonal[row] = 1
            sparse_matrix.append((val,row,col))
        if 0 in check_diagonal and condition == True:
            print("Diagonala principala nu contine toate elementele nenule!")
            return None
    return n,sparse_matrix

def read_B_from_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        n = int(lines[0])
        b = [0] * n
        contor = 0
        for line in lines[1:]:
            val = float(line)
            b[contor] = val
            contor += 1
    return n,b

def memorize_matrix_1(n, sparse_matrix):
    values = []
    ind_col = []
    start_lines = [0] * (n+1)
    sparse_matrix.sort(key=lambda x: (x[1], x[2]))

    contor = 0
    start_lines[0] = 0
    row_2 = sparse_matrix[0][1]
    last_col = 0
    for val,row,col in sparse_matrix:
        contor += 1
        values.append(val)
        ind_col.append(col)
        if row_2 != row:
            start_lines[row] = copy.deepcopy(contor-1)
        elif row_2 == row and last_col == col and contor!=1:
            print("same row and col")
            print(f"row_2 = {row_2}, last_col = {last_col}")
            values.pop()
            last_value = values.pop()
            values.append(last_value + val)
            ind_col.pop()
        row_2 = copy.deepcopy(row)
        last_col = copy.deepcopy(col)
    start_lines[n] = len(sparse_matrix)
    ind_col.append(contor-1)
    return values,ind_col,start_lines

def memorize_matrix_2(n, sparse_matrix):
    sparse_matrix.sort(key = lambda x: (x[1],x[2]))
    values = []
    row_indices = []
    col_indices = []
    for val,row, col in sparse_matrix:
        last_row = row_indices[-1] if row_indices else None
        last_col = col_indices[-1] if col_indices else None
        if row == last_row and col == last_col:
            last_values = values.pop()
            values.append(last_values + val)
        else:
            row_indices.append(row)
            col_indices.append(col)
            values.append(val)
    return n, values, row_indices, col_indices

def Gauss_Seidel_2(n, values, row, col, b):
    x_gs = [0] * n
    kmax = 10000
    k = 0
    print(f"Valori: {values[0:10]}")
    while k < kmax:
        delta_x = 0
        print(f"Iteratia {k}")
        start = 0

        for i in range(n):
            sum1 = sum2 = 0
            diagonal = 1
            for j in range(start,len(values)):
                if(row[j]>i):
                    start = j
                    break
                if row[j] == i and col[j] < i:
                    sum1 += values[j] * x_gs[col[j]]
                if row[j] == i and col[j] == i:
                    diagonal = values[j]
                if row[j] == i and col[j] > i:
                    sum2 += values[j] * x_gs[col[j]]

            x_new = (b[i] - sum1 - sum2) / diagonal
            delta_x = max(delta_x,abs(x_new - x_gs[i]))
            x_gs[i] = x_new


        print(f"Diferenta maxima {k} : {delta_x}")
        if delta_x < epsilon:
            return x_gs,k  # Soluția a convergat

        if delta_x >= 10**8 and k > 1:
            print(f"Delta_x = {delta_x} >= 10^8 for k = {k}!")
            return x_gs, k

        k += 1

    print("Nu a convergat în {} iterații.".format(kmax))
    return None,None

def Gauss_Seidel_1(n, values, ind_col, start_lines, b):
    x_gs = [0] * n
    kmax = 10000
    k = 0

    while k < kmax:
        delta_x = 0

        for i in range(n):
            sum1 = sum2 = 0
            diagonal = 1
            start_index = start_lines[i]
            end_index = start_lines[i + 1]

            for j in range(start_index, end_index):
                col_index = ind_col[j]
                if col_index < i:
                    sum1 += values[j] * x_gs[col_index]
                elif col_index > i:
                    sum2 += values[j] * x_gs[col_index]
                else:
                    diagonal = values[j]

            x_new = (b[i] - sum1 - sum2) / diagonal
            delta_x = max(delta_x, abs(x_new - x_gs[i]))

            x_gs[i] = x_new

        print(f"Diferenta maxima {k} : {delta_x}")

        if delta_x < epsilon:
            return x_gs, k

        if delta_x >= 10**8 and k > 1:
            print("Solutia nu converge!")
            print(f"Delta_x = {delta_x} >= 10^8 for k = {k}!")
            return x_gs, k

        k += 1

    print("Nu a convergat în {} iterații.".format(kmax))
    return None, None

def calculate_rezidual(sparse_matrix, x, b):
    rezidual = [0] * len(b)
    for val, row, col in sparse_matrix:
        rezidual[row] += val * x[col]
    for i in range(len(b)):
        rezidual[i] = b[i] - rezidual[i]
    return np.linalg.norm(rezidual)


def rezolve_for_file_i_1(i):
    file_name_A = f"a_{i}.txt"
    n, sparse_matrix = read_A_from_file(file_name_A)
    sparse_matrix.sort(key=lambda x: (x[1], x[2]))
    values, ind_col, start_lines = memorize_matrix_1(n, sparse_matrix)
    file_name_B = f"b_{i}.txt"
    n, b = read_B_from_file(file_name_B)
    if n is not None and b is not None:
        x, iteratii = Gauss_Seidel_1(n, values, ind_col, start_lines, b)
        if x is not None:
            print(f"Rezolvare pentru fișierele {file_name_A} și {file_name_B}: {x[1:10]}")
            print("Iteratii: ", iteratii)
            print("Rezidual: ", calculate_rezidual(sparse_matrix, x, b))
        else:
            print("Solutia nu converge!")
    else:
        print("Datele de intrare nu sunt valide!")

def rezolve_for_file_i_2(i):
    file_name_A = f"a_{i}.txt"
    n, sparse_matrix = read_A_from_file(file_name_A)
    n, values, row_indices, col_indices = memorize_matrix_2(n, sparse_matrix)
    file_name_B = f"b_{i}.txt"
    n, b = read_B_from_file(file_name_B)
    if n is not None and b is not None:
        x, iteratii = Gauss_Seidel_2(n, values, row_indices, col_indices, b)
        if x is not None:
            print(f"Rezolvare pentru fișierele {file_name_A} și {file_name_B}: {x[1:10]}")
            print("Iteratii: ", iteratii)
            print("Rezidual: ", calculate_rezidual(sparse_matrix, x, b))
        else:
            print("Solutia nu converge!")
    else:
        print("Datele de intrare nu sunt valide!")

def sum_matrix_2(n, values_A, row_indices_A, col_indices_A, values_B, row_indices_B, col_indices_B):
    values_C = []
    row_indices_C = []
    col_indices_C = []
    while values_A != [] and values_B != []:
        if row_indices_A[0] < row_indices_B[0] or (row_indices_A[0] == row_indices_B[0] and col_indices_A[0] < col_indices_B[0]):
            values_C.append(values_A.pop(0))
            row_indices_C.append(row_indices_A.pop(0))
            col_indices_C.append(col_indices_A.pop(0))
        elif row_indices_A[0] > row_indices_B[0] or (row_indices_A[0] == row_indices_B[0] and col_indices_A[0] > col_indices_B[0]):
            values_C.append(values_B.pop(0))
            row_indices_C.append(row_indices_B.pop(0))
            col_indices_C.append(col_indices_B.pop(0))
        else:
            if values_A[0] + values_B[0] != 0:
                values_C.append(values_A.pop(0) + values_B.pop(0))
                row_indices_C.append(row_indices_A.pop(0))
                col_indices_C.append(col_indices_A.pop(0))
                col_indices_B.pop(0)
                row_indices_B.pop(0)
            else:
                values_A.pop(0)
                values_B.pop(0)
                row_indices_A.pop(0)
                row_indices_B.pop(0)
                col_indices_A.pop(0)
                col_indices_B.pop(0)
    return n, values_C, row_indices_C, col_indices_C


if __name__ == '__main__':
    epsilon = 10 ** (-set_machine_precision())

    print("************ Tema  ************")
    for i in range(1, 6):
        print(f"Rezolvare pentru fișierele a_{i}.txt și b_{i}.txt")
        rezolve_for_file_i_1(i)
        print("----------------------------------------")
        rezolve_for_file_i_2(i)
        print("\n\n")

    print("************ Bonus ************")
    n, sparse_matrix_A = read_A_from_file("a.txt")
    n, values_A, row_indices_A, col_indices_A = memorize_matrix_2(n, sparse_matrix_A)
    n, sparse_matrix_B = read_A_from_file("b.txt")
    n, values_B, row_indices_B, col_indices_B = memorize_matrix_2(n, sparse_matrix_B)
    n, values_C, row_indices_C, col_indices_C = sum_matrix_2(n, values_A, row_indices_A, col_indices_A, values_B, row_indices_B, col_indices_B)
    n, sparse_matrix_A_plus_B = read_A_from_file("aplusb.txt", False)
    n, values_A_plus_B, row_indices_A_plus_B, col_indices_A_plus_B = memorize_matrix_2(n, sparse_matrix_A_plus_B)
    if len(values_C) != len(values_A_plus_B):
        print("Suma nu este corectă! Lungimi diferite!")
    else:
        for i in range(len(values_C)):
            if math.fabs(values_C[i] - values_A_plus_B[i]) > epsilon:
                print("Suma nu este corectă!")
                print(f"values_C[{i}] = {values_C[i]}, values_A_plus_B[{i}] = {values_A_plus_B[i]}")
                print(f"row_indices_C[{i}] = {row_indices_C[i]}, row_indices_A_plus_B[{i}] = {row_indices_A_plus_B[i]}")
                print(f"col_indices_C[{i}] = {col_indices_C[i]}, col_indices_A_plus_B[{i}] = {col_indices_A_plus_B[i]}")
                break
        else:
            print("Suma este corectă!")
            print(values_C[1:10])
            print(values_A_plus_B[1:10])
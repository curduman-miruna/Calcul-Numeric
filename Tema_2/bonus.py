from main import set_machine_precision, verify_division_by_epsilon


def crout_decomposition_LU(A):
    n = len(A)
    L = [0.0] * (n * (n + 1) // 2)
    U = [0.0] * (n * (n + 1) // 2)

    index_L = 0
    index_U = 0

    for j in range(n):
        for i in range(j, n):
            sum_L = 0.0
            sum_U = 0.0
            for k in range(j):
                sum_L += L[i * (i + 1) // 2 + k] * U[k * (k + 1) // 2 + j]
                sum_U += L[j * (j + 1) // 2 + k] * U[k * (k + 1) // 2 + i]
            L[i * (i + 1) // 2 + j] = A[i][j] - sum_L if i >= j else 0.0
            U[j * (j + 1) // 2 + i] = (A[i][j] - sum_U) / L[j * (j + 1) // 2 + j] if i >= j else A[i][j]

    return L, U


def forward_substitution(L, b):
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i * (i + 1) // 2 + j] * y[j]
    return y

def backward_substitution(U, y):
    n = len(y)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[j * (j + 1) // 2 + i] * x[j]
        x[i] /= U[i * (i + 1) // 2 + i]
    return x


b = [2, 2, 2]
A = [[2.5, 2, 2], [5, 6, 5], [5, 6, 6.5]]
L, U = crout_decomposition_LU(A)

y = forward_substitution(L, b)
x = backward_substitution(U, y)

print("Solution:", x)
print("L:")
for i in range(len(A)):
    print(L[i * (i + 1) // 2: (i + 1) * (i + 2) // 2])

print("U:")
for i in range(len(A)):
    print(U[i * (i + 1) // 2: (i + 1) * (i + 2) // 2])

print("X:",x)
import math
import tkinter as tk
import random
def T(i,a):
    match i:
        case 1:
            return a
        case 2:
            return 3*a/(3-a**2)
        case 3:
            return (15*a-a**3)/(15-6*a**2)
        case 4:
            return (105*a-10*a**3)/(105-105*a**2+15*a**4)
        case 5:
            return (945*a-105*a**3+a**5)/(945-420*a**2+15*a**4)
        case 6:
            return (10395*a-1260*a**3+21*a**5)/(10395-4725*a**2+210*a**4-a**6)
        case 7:
            return (135135*a-17325*a**3+378*a**5-a**7)/(135135-62370*a**2+3150*a**4-28*a**6)
        case 8:
            return (2027025*a-270270*a**3+6930*a**5-36*a**7)/(2027025-945945*a**2+51975*a**4-630*a**6+a**8)
        case 9:
            return (34459425*a-4729725*a**3+135135*a**5-990*a**7+a**9)/(34459425-16216200*a**2+945945*a**4-13860*a**6+45*a**8)
        case _:
            raise ValueError(f"Invalid value of i: {i}")

def S(i,a):
    return (1-T(i,(2*a-math.pi)/4)**2) / (1+T(i,(2*a-math.pi)/4)**2)

def C(i,a):
    return (1-T(i,(a/2))**2) / (1+T(i,(a/2))**2)

def t_aprox():
    error_function = [[0, 0] for _ in range(6)]
    for i in range(4, 10):
        for _ in range(1, 10001):
            random_number = random.uniform(-math.pi / 2, math.pi / 2)
            error_function[i - 4][0] += abs(T(i, random_number) - math.tan(random_number))
            error_function[i - 4][1] = i
        print(f"Error for T({i}) is: {error_function[i - 4][0]}")
    print("\nTop 3 approximations for T(i) are:")
    sorted_error_function = sorted(error_function, key=lambda x: x[0])
    print(sorted_error_function[0][0])
    for i in range (0,3):
        print(f"Approximation for T({sorted_error_function[i][1]}) is: {sorted_error_function[i][0]}")

def sin_aprox():
    error_function = [[0, 0] for _ in range(6)]
    for i in range(4, 10):
        for _ in range(1, 10001):
            random_number = random.uniform(-math.pi, math.pi)
            error_function[i - 4][0] += abs(S(i, random_number) - math.sin(random_number))
            error_function[i - 4][1] = i
        print(f"Error for S({i}) is: {error_function[i - 4][0]}")
    print("\nTop 3 approximations for S(i) are:")
    sorted_error_function = sorted(error_function, key=lambda x: x[0])
    for i in range (0,3):
        print(f"Approximation for S({sorted_error_function[i][1]}) is: {sorted_error_function[i][0]}")

def cos_aprox():
    error_function = [[0, 0] for _ in range(6)]
    for i in range(4, 10):
        for _ in range(1, 10001):
            random_number = random.uniform(-math.pi, math.pi)
            error_function[i - 4][0] += abs(C(i, random_number) - math.cos(random_number))
            error_function[i - 4][1] = i
        print(f"Error for C({i}) is: {error_function[i - 4][0]}")
    print("\nTop 3 approximations for C(i) are:")
    sorted_error_function = sorted(error_function, key=lambda x: x[0])
    for i in range (0,3):
        print(f"Approximation for C({sorted_error_function[i][1]}) is: {sorted_error_function[i][0]}")


t_aprox()
print("\n")
sin_aprox()
print("\n")
cos_aprox()




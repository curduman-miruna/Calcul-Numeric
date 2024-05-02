import random

import sympy

epsilon = 10 ** -16
niu = 10 ** -5
def set_machine_precision():
    t = int(input("Introduceti pentru care 10^(-t) este precizia masinii: "))
    return t

def verify_epsilon(number):
    if abs(number) <= epsilon:
        print(f"Nu se poate imparti, număr = {number} mai mic decât epsilon = {epsilon}!")
        return False
    return True


def first_derivative(f, x, y, h=1e-6):
    G1 = (3*f(x + h, y) - 4*f(x - h, y)+f(x-2*h,y)) / (2 * h)
    G2 = (3*f(x, y + h) - 4*f(x, y - h)+f(x,y-2*h)) / (2 * h)
    return G1, G2

def second_derivative(f, x, y, h=1e-6):
    d2f_dx2 = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h ** 2
    d2f_dy2 = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h ** 2
    return d2f_dx2, d2f_dy2

def modul_derivative(F, x, y):
    return (first_derivative(F, x, y)[0] ** 2 + first_derivative(F, x, y)[1] ** 2)
def calculate_niu(F,x,y):
    niu = 1
    p = 1
    beta = 0.8
    x_derivative = first_derivative(F, x, y)[0]
    y_derivative = first_derivative(F, x, y)[1]
    while (F(x-x_derivative,y-y_derivative) > F(x,y) - niu/2 * modul_derivative(F,x,y) ) & p < 8:
        niu = beta * niu
        p += 1

    return niu

def main():
    k = 0
    F, x0, y0 = initialize_data_random()
    niu = calculate_niu(F, x0, y0)
    x = x0 - niu * first_derivative(F, x0, y0)[0]
    y = y0 - niu * first_derivative(F, x0, y0)[1]
    k += 1
    while verify_epsilon(niu * modul_derivative(F, x, y)) & (k<30000) & (niu * modul_derivative(F, x, y) < 10**10):
        x = x - niu * first_derivative(F, x, y)[0]
        y = y - niu * first_derivative(F, x, y)[1]
        niu = calculate_niu(F, x, y)
        k += 1
    if(niu * modul_derivative(F, x, y) > 10**10):
        print("Nu se poate imparti, număr = {niu * modul_derivative(F, x, y)} mai mare decât epsilon = {epsilon}!")
        return
    print(f"Solutia este: {x}, {y}")

def initialize_data_random():
    F = lambda x,y: x**2 + y**2 -2*x -4*y -1
    x0 = random.randint(1, 100)
    y0 = random.randint(1, 100)
    return F, x0, y0

if __name__ == "__main__":
    main()
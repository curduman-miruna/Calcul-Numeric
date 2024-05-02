import copy
import math
import random
import matplotlib.pyplot as plot
import numpy
import numpy as np

epsilon = 10 ** -16

def initialize_data_random():
    n = int(input("Introduceti n: "))
    x0 = int(input("Introduceti x0: "))
    xn = int(input("Introduceti xn: "))
    if xn < x0:
        xn = int(input("Introduceti xn (mai mic decat x0) : "))
    h = (xn - x0) / n
    x_values = [x0 + i * h for i in range(n+1)]
    f = lambda x: x ** 3+ 5 * x ** 2 - 10 * x + 23
    y_values = [f(x) for x in x_values]
    return x_values, y_values, h, f

def initialize_data_for_newt():
    x_values = [0,1,2,3,4,5]
    y_values = [50,47,-2,-121,-310,-545]
    h = (5-0) / 5
    x_aprox = 1.5
    y_aprox = 30.3125
    return x_values, y_values, h, x_aprox, y_aprox

def initialize_data_for_newt_2():
    x_values = [0,2,4,6]
    y_values = [1,-3,49,253]
    h = (6-0) / 3
    x_aprox = 1
    y_aprox = -2
    return x_values, y_values, h, x_aprox, y_aprox

def initiliaze_data_for_square_met():
    x0 = 1
    a = 1
    xn = 5
    b = 5
    f = lambda x: x ** 4 - 12 * x ** 3 + 30 * x ** 2 + 12
    n = 25
    x_values = [0.0] * (n + 1)
    x_values[0] = x0
    x_values[n] = xn
    h = (xn - x0) / n
    for i in range(1, n):
        x_values[i] = x0 + i * h
    y_values = [f(x) for x in x_values]
    m = 4
    return x_values, y_values, a, b, f, m, n

def newton_first_factor(n,y):
    delta_temp = []
    delta_x0 = []
    delta_x0.append(y[1]-y[0])
    for i in range(1,n):
        if i == 1:
            for j in range(0,n-i):
                delta_temp.append(y[j+1]-y[j])
        else:
            delta_x0.append(delta_temp[1]-delta_temp[0])
            delta_new = []
            for j in range(0,n-i):
                delta_new.append(delta_temp[j+1]-delta_temp[j])
            delta_temp = copy.deepcopy(delta_new)
    return delta_x0


def newton_second_factor(n,t):
    s = []
    s.append(t)
    for k in range(2, n + 1):
        last_value = s[-1]
        s.append(last_value * (t - k + 1) / k)
    return s

def newton_interpolation(x_values, y_values, h, x):
    n = len(x_values)
    t = (x - x_values[0]) / h
    delta_y = newton_first_factor(n, y_values)
    s = newton_second_factor(n,t)
    result = y_values[0]
    for i in range(1, n):
        result += s[i-1] * delta_y[i-1]
    return result

def smallest_square_method(x_aprox, x, y, n, m):
    B = [[0 for i in range(m + 1)] for j in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            suma = 0
            for k in range(n + 1):
                suma += x[k] ** (i + j)
            B[i][j] = suma

    f = [0 for i in range(m + 1)]
    for i in range(m + 1):
        suma = 0
        for k in range(n + 1):
            suma += (x[k] ** i) * y[k]
        f[i] = suma

    # rezolvarea sistemului liniar B*a = f
    a = numpy.linalg.solve(B, f)
    f_de_x_aprox = horner_method(a, x_aprox)
    return f_de_x_aprox
def horner_method(polinom, x):
    rezultat = polinom[0]
    for i in range(1, len(polinom)):
        rezultat = rezultat * x + polinom[i]
    return rezultat

if __name__ == '__main__':
    x_values_for_newt, y_values_for_newt, h, x_aprox_for_newt, y_aprox_for_newt = initialize_data_for_newt()
    #Calculation for Newton's progressive interpolation
    print("---------------- Newton's progressive interpolation using Aitken schema ----------------")
    print (f"x_values_for_newt = {x_values_for_newt}")
    print (f"y_values_for_newt = {y_values_for_newt}")
    print(f"Newton's progressive interpolation for x = {x_aprox_for_newt} is {newton_interpolation(x_values_for_newt, y_values_for_newt, h, x_aprox_for_newt)}")
    print(f"Norm for aproximation = {abs(y_aprox_for_newt - newton_interpolation(x_values_for_newt, y_values_for_newt, h, x_aprox_for_newt))}")
    print()

    #Calculation using small square method
    print("---------------- Small square method ----------------")
    x_values_sqr, y_values_sqr, a, b, f, m, n = initiliaze_data_for_square_met()
    x_aprox_sqr = random.uniform(a, b)
    print(f"x_values_sqr = {x_values_sqr}")
    print(f"y_values_sqr = {y_values_sqr}")
    print(f"x_aprox_sqr = {x_aprox_sqr}")
    y_aprox_sqr = smallest_square_method(x_aprox_sqr, x_values_sqr, y_values_sqr, n, m)
    print(f"Small square method for x = {x_aprox_sqr} is {y_aprox_sqr}")
    print(f"Norm for aproximation = {abs(f(x_aprox_sqr) - y_aprox_sqr)}")
    Norm_sum = 0
    y_smallest_square = [smallest_square_method(x, x_values_sqr, y_values_sqr, n, m) for x in x_values_sqr]
    print(f"y_smallest_square = {y_smallest_square}")
    for i in range(n+1):
        Norm_sum += math.fabs(y_values_sqr[i] - y_smallest_square[i])
    print(f"Sum of norm for aproximation = {Norm_sum}")

    #Bonus
    x_values_for_newt_aprox = np.arange(x_values_for_newt[0], x_values_for_newt[-1], 0.01)
    y_values_for_newt_aprox = [newton_interpolation(x_values_for_newt, y_values_for_newt, h, x) for x in x_values_for_newt_aprox]
    plot.plot(x_values_for_newt_aprox, y_values_for_newt_aprox, label="Newton's progressive interpolation")

    plot.plot(x_values_sqr, y_values_sqr, label="Function f(x) = x^4 - 12 * x^3 + 30 * x^2 + 12")
    plot.plot(x_values_sqr, y_smallest_square, label="Small square method aproximation")

    plot.xlabel('x - axis')
    plot.ylabel('y - axis')
    plot.title('The graphs aproximation')
    plot.legend()
    plot.grid(True)
    plot.show()


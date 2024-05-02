import math
import tkinter as tk
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

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
    return error_function

def sin_aprox():
    error_function = [[0, 0] for _ in range(6)]
    for i in range(4, 10):
        for _ in range(1, 10001):
            random_number = random.uniform(-math.pi, math.pi)
            error_function[i - 4][0] += abs(S(i, random_number) - math.sin(random_number))
            error_function[i - 4][1] = i
    return error_function

def cos_aprox():
    error_function = [[0, 0] for _ in range(6)]
    for i in range(4, 10):
        for _ in range(1, 10001):
            random_number = random.uniform(-math.pi, math.pi)
            error_function[i - 4][0] += abs(C(i, random_number) - math.cos(random_number))
            error_function[i - 4][1] = i
    return error_function

def afiseaza_grafic_T():
    plt.figure(figsize=(8, 6))
    error_function_t = t_aprox()
    x = np.arange(4, 10)
    y_t = [error[0] / 10000 for error in error_function_t]
    plt.plot(x, y_t, marker='o', linestyle='-', color='b')
    plt.title('Aproximarea erorii pentru funcția T')
    plt.xlabel('i')
    plt.ylabel('Eroare medie')
    plt.grid(True)
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

def afiseaza_grafic_S():
    plt.figure(figsize=(8, 6))
    error_function_s = sin_aprox()
    x = np.arange(4, 10)
    y_s = [error[0] / 10000 for error in error_function_s]
    plt.plot(x, y_s, marker='o', linestyle='-', color='r')
    plt.title('Aproximarea erorii pentru funcția S')
    plt.xlabel('i')
    plt.ylabel('Eroare medie')
    plt.grid(True)
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

def afiseaza_grafic_C():
    plt.figure(figsize=(8, 6))
    error_function_c = cos_aprox()
    x = np.arange(4, 10)
    y_c = [error[0] / 10000 for error in error_function_c]
    plt.plot(x, y_c, marker='o', linestyle='-', color='g')
    plt.title('Aproximarea erorii pentru funcția C')
    plt.xlabel('i')
    plt.ylabel('Eroare medie')
    plt.grid(True)
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

root = tk.Tk()
root.title("Grafice pentru erorile lui T, S și C")

frame = tk.Frame(root)
frame.pack(pady=10)

btn_t = tk.Button(root, text="Grafic T", command=afiseaza_grafic_T)
btn_t.pack(side=tk.LEFT, padx=5)

btn_s = tk.Button(root, text="Grafic S", command=afiseaza_grafic_S)
btn_s.pack(side=tk.LEFT, padx=5)

btn_c = tk.Button(root, text="Grafic C", command=afiseaza_grafic_C)
btn_c.pack(side=tk.LEFT, padx=5)

root.mainloop()

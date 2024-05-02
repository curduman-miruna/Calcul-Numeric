import tkinter as tk
from tkinter import ttk
import math
import random

epsilon = 10 ** -16


def verify_epsilon(number):
    if abs(number) <= epsilon:
        return False
    return True


def horner_method(polinom, x):
    rezultat = polinom[0]
    for i in range(1, len(polinom)):
        rezultat = rezultat * x + polinom[i]
    return rezultat


def set_the_interval(a1):
    A = max(a1)
    R = (abs(a1[0]) + A) // abs(a1[0])
    return R


def muller_method(pol, R, n):
    xk = random.uniform(int(-R), int(R))
    xk_1 = random.uniform(-R, R)
    if xk == xk_1:
        xk_1 = random.uniform(-R, R)
    xk_2 = random.uniform(-R, R)
    if xk_2 == xk or xk_2 == xk_1:
        xk_2 = random.uniform(-R, R)
    delta_x = 0.1
    k = 3
    kmax = 1000
    while verify_epsilon(delta_x) and k < kmax and math.fabs(delta_x) <= 10 ** 8:
        h0 = xk_1 - xk_2
        h1 = xk - xk_1
        if verify_epsilon(h0) == False or verify_epsilon(h1) == False:
            print("Line 44")
            muller_method(pol, R, n)
            return
        if verify_epsilon(h0 + h1) == False:
            print("Line 46")
            muller_method(pol, R, n)
            return
        d0 = (horner_method(pol, xk_1) - horner_method(pol, xk_2)) / h0
        d1 = (horner_method(pol, xk) - horner_method(pol, xk_1)) / h1
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = horner_method(pol, xk)
        if b ** 2 - 4 * a * c < 0:
            print("Delta negativ")
            muller_method(pol, R, n)
            return
        if not verify_epsilon(b + (-1 if b <= 0 else 1) * math.sqrt(b ** 2 - 4 * a * c)):
            print("Line 58")
            muller_method(pol, R, n)
            return
        delta_x = 2 * c / (b + (-1 if b <= 0 else 1) * math.sqrt(b ** 2 - 4 * a * c))
        xk1 = xk - delta_x
        k += 1
        xk = xk_1
        xk_1 = xk_2
        xk_2 = xk1

    if verify_epsilon(delta_x) == False:
        return xk_2
    else:
        print("Divergenta")
        muller_method(pol, R, n)
        return


def solve_for_a(coeficienti, coeficienti_f, tab):
    n = len(coeficienti) - 1
    R = set_the_interval(coeficienti)
    x_aprox = muller_method(coeficienti, R, n)
    while x_aprox is None:
        x_aprox = muller_method(coeficienti, R, n)
    print(f"Solutia este: {x_aprox}")
    label1 = ttk.Label(tab, text=f"Solutia este: {x_aprox}")
    label1.pack()
    with open(f"muller_{coeficienti_f}.txt", "r") as file:
        if str(x_aprox) in file.read():
            label3 = ttk.Label(tab, text="Solutia este in fisier deja.")
            label3.pack()
            return
    with open(f"muller_{coeficienti_f}.txt", "a") as file:
        file.write(str(x_aprox) + "\n")
        print("Solutia a fost scrisa in fisier")
        label4 = ttk.Label(tab, text="Solutia a fost scrisa in fisier.")
        label4.pack()

a2 = [1, -6, 11, -6]
a3 = [42,-55,-42,49,-6]
a1 = [8,-38,49,-22,3]

def calculate(name, tab):
    if name == "a1":
        solve_for_a(a1, "a1", tab)
    elif name == "a2":
        solve_for_a(a2, "a2", tab)
    elif name == "a3":
        solve_for_a(a3, "a3", tab)


def create_tab(tab_control, name):
    tab = ttk.Frame(tab_control)
    tab_control.add(tab, text=name)
    button_calculate = ttk.Button(tab, text="Calculate", command=lambda: calculate(name, tab))
    button_calculate.pack()
def create_gui():
    root = tk.Tk()
    root.geometry("300x300")
    root.title("Tema 7")
    tab_control = ttk.Notebook(root)
    tab_control.pack(expand=1, fill='both')

    tab_names = ["a1", "a2", "a3"]
    for name in tab_names:
        create_tab(tab_control, name)

    root.mainloop()


if __name__ == "__main__":
    create_gui()

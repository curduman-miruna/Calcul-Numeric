import tkinter as tk


def precizia_masina():
    m = 0
    u = 10 ** (-m)
    while 1 + u != 1:
        m = m + 1
        u = 10 ** (-m)
    return m-1


def afisare():
    m = precizia_masina()
    rezultat_label.config(text="Valoarea lui m este: " + str(m))


# fereastra
root = tk.Tk()
root.title("Precizie masina")
root.geometry("400x200")

# buton
calculeaza_button = tk.Button(root, text="Calculează precizia", command=afisare)
calculeaza_button.pack(pady=30)

# afisare rezultat
rezultat_label = tk.Label(root, text="", font=("Helvetica", 14))
rezultat_label.pack(pady=5)

# bucla interfata
root.mainloop()

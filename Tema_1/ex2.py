import tkinter as tk


def precizia_masina_adunare():
    m = 0
    u = 10 ** (-m)
    while 1 + u != 1:
        m = m + 1
        u = 10 ** (-m)
    return 10 ** (-m+1)


def adunare_neasoc():
    x = 1.0
    y = precizia_masina_adunare()/10
    z = precizia_masina_adunare()/10
    return (x + y) + z != x + (y + z)


def precizie_masina_inmultire():
    m = 0
    u = 10 ** (-m)
    while u * u != 0:
        m = m + 1
        u = 10 ** (-m)
    return -m


def inmultire_neasoc():
    x = 100
    y = 10 ** precizie_masina_inmultire()
    z = 10 ** precizie_masina_inmultire()
    return (x * y) * z != x * (y * z)


def afisare1():
    verif_adunare = (adunare_neasoc())
    rezultat1_label.config(text=str(verif_adunare))


def afisare2():
    verif_inmultire = (inmultire_neasoc())
    rezultat2_label.config(text=str(verif_inmultire))


# fereastra
root = tk.Tk()
root.title("Precizie masina")
root.geometry("400x400")

# buton
calculeaza_button1 = tk.Button(root, text="Exemplu neasociativ (adunare)?", command=afisare1)
calculeaza_button1.pack(pady=30)

# afisare rezultat1
rezultat1_label = tk.Label(root, text="", font=("Helvetica", 14))
rezultat1_label.pack(pady=5)

# buton
calculeaza_button2 = tk.Button(root, text="Exemplu neasociativ (inmultire)?", command=afisare2)
calculeaza_button2.pack(pady=30)

# afisare rezultat2
rezultat2_label = tk.Label(root, text="", font=("Helvetica", 14))
rezultat2_label.pack(pady=5)

# bucla interfata
root.mainloop()


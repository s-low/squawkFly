#!/usr/local/bin/python
from Tkinter import *
import ttk
from tkFileDialog import askopenfilename

root = Tk()
root.title("squawkFly")


def choose1():
    filename = askopenfilename()
    calib1.set(filename)


def choose2():
    filename = askopenfilename()
    clip1.set(filename)


def choose3():
    filename = askopenfilename()
    calib2.set(filename)


def choose4():
    filename = askopenfilename()
    clip2.set(filename)


def submit(*args):
    print "--Submit--"
    print calib1.get()
    print calib2.get()
    print clip1.get()
    print clip2.get()

# try:
# value = float(feet.get())
# meters.set((0.3048 * value * 10000.0 + 0.5) / 10000.0)
# except ValueError:
# pass

frame = ttk.Frame(root, padding="3 3 12 12")
frame.grid(column=0, row=0, sticky=(N, W, E, S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

calib1 = StringVar()
calib2 = StringVar()
clip1 = StringVar()
clip2 = StringVar()

calib1_entry = ttk.Entry(frame, width=15, textvariable=calib1)
clip1_entry = ttk.Entry(frame, width=15, textvariable=clip1)

calib2_entry = ttk.Entry(frame, width=15, textvariable=calib2)
clip2_entry = ttk.Entry(frame, width=15, textvariable=clip2)

calib1_entry.grid(column=2, row=1, sticky=(W, E))
clip1_entry.grid(column=2, row=2, sticky=(W, E))

calib2_entry.grid(column=2, row=3, sticky=(W, E))
clip2_entry.grid(column=2, row=4, sticky=(W, E))

# FILE EXPLORER BUTTONS
choose1 = ttk.Button(frame, text="Choose", command=choose1)
choose1.grid(column=3, row=1, sticky=(W, E))

choose2 = ttk.Button(frame, text="Choose", command=choose2)
choose2.grid(column=3, row=2, sticky=(W, E))

choose3 = ttk.Button(frame, text="Choose", command=choose3)
choose3.grid(column=3, row=3, sticky=(W, E))

choose4 = ttk.Button(frame, text="Choose", command=choose4)
choose4.grid(column=3, row=4, sticky=(W, E))

# ANALYSE BUTTON
button = ttk.Button(frame, text="Analyse", command=submit)
button.grid(column=2, row=5, sticky=(W, E))


ttk.Label(frame, text="Calibration Video 1").grid(column=1, row=1, sticky=E)
ttk.Label(frame, text="FK Video 1").grid(column=1, row=2, sticky=E)
ttk.Label(frame, text="Calibration Video 2").grid(column=1, row=3, sticky=E)
ttk.Label(frame, text="FK Video 2").grid(column=1, row=4, sticky=E)

for child in frame.winfo_children():
    child.grid_configure(padx=5, pady=5)

calib1_entry.focus()
root.bind('<Return>', submit)

root.mainloop()

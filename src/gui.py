#!/usr/local/bin/python
from Tkinter import *
import ttk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import os

root = Tk()
root.title("squawkFly")


def handleSpaces(string):
    i = string.find(' ')
    new = string[:i] + '\\' + string[i:]

    i = new.find(' ')
    if new.find('\ ') != i - 1:
        new = handleSpaces(new)

    return new


def choose1():
    filename = askdirectory()
    calib1.set(filename)


def choose2():
    filename = askdirectory()
    clip1.set(filename)


def choose3():
    filename = askdirectory()
    calib2.set(filename)


def choose4():
    filename = askdirectory()
    clip2.set(filename)


def submit(*args):
    print "--Submit--"
    # escape any spaces in the path
    cal1 = handleSpaces(calib1.get())
    cal2 = handleSpaces(calib2.get())
    # vid1 = handleSpaces(clip1.get())
    # vid2 = handleSpaces(clip2.get())

    os.system("./calibrate.py " + cal1)
    os.system("./calibrate.py " + cal2)


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

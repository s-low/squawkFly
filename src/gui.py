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

    cal1 = calib1.get()
    cal2 = calib2.get()
    vid1 = clip1.get()
    vid2 = clip2.get()

    if not cal1 or not cal2 or not vid1 or not vid2:
        print "WARN: Please select all four files"
        return

    # escape any spaces in the path
    cal1 = handleSpaces(calib1.get())
    cal2 = handleSpaces(calib2.get())
    vid1 = handleSpaces(clip1.get())
    vid2 = handleSpaces(clip2.get())

    os.system("./calibrate.py " + cal1 + ' user/camera1.txt')
    os.system("./calibrate.py " + cal2 + ' user/camera2.txt')
    os.system("./postPoints.py " + vid1 + ' user/postPts1.txt')
    os.system("./postPoints.py " + vid2 + ' user/postPts2.txt')
    os.system("./detect.py " + vid1 + ' user/detections1.txt')
    os.system("./detect.py " + vid2 + ' user/detections2.txt')
    os.system("./kalman.py user/detections1.txt user/trajectories1.txt")
    os.system("./kalman.py user/detections2.txt user/trajectories2.txt")
    os.system("./trajectories.py -1 user/detections1.txt \
        user/trajectories1.txt user/trajectory1.txt")
    os.system("./trajectories.py -1 user/detections2.txt \
        user/trajectories2.txt user/trajectory2.txt")
    os.system("./interpolate.py user/trajectory1.txt 30")
    os.system("./interpolate.py user/trajectory2.txt 30")


frame = ttk.Frame(root, padding="3 3 12 12")
frame.grid(column=0, row=0, sticky=(N, W, E, S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

calib1 = StringVar()
calib2 = StringVar()
clip1 = StringVar()
clip2 = StringVar()

calib1.set('/Users/samlow/Google Drive/res/lumix')
calib2.set('/Users/samlow/Google Drive/res/g3')
clip1.set('/Users/samlow/Google Drive/res/coombe/clips/crossbar/lumix')
clip2.set('/Users/samlow/Google Drive/res/coombe/clips/crossbar/g3')

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

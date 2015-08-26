#!/usr/local/bin/python
from Tkinter import *
import ttk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import os


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

    session = session_name.get()
    clip = clip_name.get()
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

    # paths
    p_session = "sessions/" + session
    p_clip = p_session + '/' + clip

    print "Session:", p_session
    print "Clip:", p_clip

    session = ' ' + p_session + '/'
    clip = ' ' + p_clip + '/'

    args_cal1 = cal1 + session + 'camera1.txt'
    args_cal2 = cal2 + session + 'camera2.txt'
    args_posts1 = vid2 + session + 'postPts1.txt' + session + 'image1.png'
    args_posts2 = vid2 + session + 'postPts2.txt' + session + 'image2.png'
    args_match = vid1 + ' ' + vid2 + session + 'statics1_.txt' + session + 'statics2_.txt'
    args_detect1 = vid1 + clip + 'detections1.txt'
    args_detect2 = vid2 + clip + 'detections2.txt'
    args_kalman1 = clip + "detections1.txt" + clip + "trajectories1.txt"
    args_kalman2 = clip + "detections2.txt" + clip + "trajectories2.txt"
    args_traj1 = clip + "detections1.txt" + clip + "trajectories1.txt" + clip + "trajectory1.txt"
    args_traj2 = clip + "detections2.txt" + clip + "trajectories2.txt" + clip + "trajectory2.txt"
    args_interp1 = clip + 'trajectory1.txt 30' + clip + 'trajectory1.txt'
    args_interp2 = clip + 'trajectory2.txt 30' + clip + 'trajectory2.txt'

    # New session, create the scene data
    if not os.path.exists(p_session):
        os.makedirs(p_session)
        os.system("./calibrate.py " + args_cal1)
        os.system("./calibrate.py " + args_cal2)
        os.system("./postPoints.py " + args_posts1)
        os.system("./postPoints.py " + args_posts2)
        os.system("./manualMatch.py " + args_match)

    # New clip, create the trajectory data
    if not os.path.exists(p_clip):
        os.makedirs(p_clip)

        os.system("./detect.py " + args_detect1)
        os.system("./detect.py " + args_detect2)
        os.system("./kalman.py" + args_kalman1)
        os.system("./kalman.py" + args_kalman2)
        os.system("./trajectories.py -1" + args_traj1)
        os.system("./trajectories.py -1" + args_traj2)
        os.system("./interpolate.py" + args_interp1)
        os.system("./interpolate.py" + args_interp2)

    else:
        print "That clip already exists"

root = Tk()
root.title("squawkFly")

frame = ttk.Frame(root, padding="3 3 12 12")
frame.grid(column=0, row=0, sticky=(N, W, E, S))

calib1 = StringVar()
calib2 = StringVar()
clip1 = StringVar()
clip2 = StringVar()
session_name = StringVar()
clip_name = StringVar()

calib1.set('/Users/samlow/Google Drive/res/lumix')
calib2.set('/Users/samlow/Google Drive/res/g3')
clip1.set('/Users/samlow/Google Drive/res/coombe/clips/crossbar/lumix')
clip2.set('/Users/samlow/Google Drive/res/coombe/clips/crossbar/g3')

session_entry = ttk.Entry(frame, width=30, textvariable=session_name)
session_entry.grid(column=2, row=1, sticky=(W, E))

clip_entry = ttk.Entry(frame, width=30, textvariable=clip_name)
clip_entry.grid(column=2, row=2, sticky=(W, E))


calib1_entry = ttk.Entry(frame, width=45, textvariable=calib1)
clip1_entry = ttk.Entry(frame, width=45, textvariable=clip1)

calib2_entry = ttk.Entry(frame, width=45, textvariable=calib2)
clip2_entry = ttk.Entry(frame, width=45, textvariable=clip2)

calib1_entry.grid(column=2, row=3, sticky=(W, E))
clip1_entry.grid(column=2, row=4, sticky=(W, E))

calib2_entry.grid(column=2, row=5, sticky=(W, E))
clip2_entry.grid(column=2, row=6, sticky=(W, E))

# FILE EXPLORER BUTTONS
choose1 = ttk.Button(frame, text="Choose", command=choose1)
choose1.grid(column=3, row=3, sticky=(W, E))

choose2 = ttk.Button(frame, text="Choose", command=choose2)
choose2.grid(column=3, row=4, sticky=(W, E))

choose3 = ttk.Button(frame, text="Choose", command=choose3)
choose3.grid(column=3, row=5, sticky=(W, E))

choose4 = ttk.Button(frame, text="Choose", command=choose4)
choose4.grid(column=3, row=6, sticky=(W, E))

# ANALYSE BUTTON
button = ttk.Button(frame, text="Analyse", command=submit)
button.grid(column=2, row=7, sticky=(W, E))

ttk.Label(frame, text="Session Name").grid(column=1, row=1, sticky=E)
ttk.Label(frame, text="Clip Name").grid(column=1, row=2, sticky=E)
ttk.Label(frame, text="Calibration Video 1").grid(column=1, row=3, sticky=E)
ttk.Label(frame, text="FK Video 1").grid(column=1, row=4, sticky=E)
ttk.Label(frame, text="Calibration Video 2").grid(column=1, row=5, sticky=E)
ttk.Label(frame, text="FK Video 2").grid(column=1, row=6, sticky=E)

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
frame.columnconfigure(2, weight=1)

for child in frame.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.update()
root.minsize(root.winfo_width(), root.winfo_height())

calib1_entry.focus()
root.bind('<Return>', submit)
root.mainloop()

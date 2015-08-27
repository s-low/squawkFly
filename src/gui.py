#!/usr/local/bin/python
from Tkinter import *
import ttk
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import os
import shutil


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


def changeClipOptions(event):
    session_value = session_name.get()
    lst = get_subdirectories('sessions/' + session_value)
    clip_entry['values'] = lst


def get_subdirectories(folder):
    return [sub for sub in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, sub))]


def delete():

    clip = clip_name.get()
    session = session_name.get()
    if session is not None and session != '':
        p_session = "sessions/" + session
        p_clip = p_session + '/' + clip

        print "Delete:", p_clip
        shutil.rmtree(p_clip)


def submit(*args):
    print "--Submit--"

    new_session = False
    new_clip = False
    videos = False
    session = session_name.get()
    clip = clip_name.get()
    cal1 = calib1.get()
    cal2 = calib2.get()
    vid1 = clip1.get()
    vid2 = clip2.get()
    view = viewer.get()

    # paths
    p_session = "sessions/" + session
    p_clip = p_session + '/' + clip

    if vid1 and vid2:
        videos = True

    # unless the session already exists, all four input fields necessary
    if not os.path.exists(p_session):
        new_session = True
        if not cal1 or not cal2 or not vid1 or not vid2:
            print "WARN: Invalid input. To create a session you must submit all four files with a valid clip name."
            return

    if not os.path.exists(p_clip):
        new_clip = True
        if not vid1 or not vid2:
            print "WARN: Invalid input. To create a new clip in an existing session you must submit both free kick videos."
            return

    if new_session:
        cal1 = handleSpaces(calib1.get())
        cal2 = handleSpaces(calib2.get())

    # technically, could supply these for the visualisation of an old clip
    if videos:
        vid1 = handleSpaces(clip1.get())
        vid2 = handleSpaces(clip2.get())

    # paths
    p_session = "sessions/" + session
    p_clip = p_session + '/' + clip

    print "Session:", p_session
    print "Clip:", p_clip

    session = ' ' + p_session + '/'
    clip = ' ' + p_clip + '/'

    args_cal1 = cal1 + session + 'camera1.txt ' + view
    args_cal2 = cal2 + session + 'camera2.txt ' + view

    args_posts1 = vid1 + session + 'postPts1.txt' + session + 'image1.png'
    args_posts2 = vid2 + session + 'postPts2.txt' + session + 'image2.png'

    args_match = vid1 + ' ' + vid2 + session + \
        'statics1.txt' + session + 'statics2.txt'

    args_detect1 = vid1 + clip + 'detections1.txt ' + view
    args_detect2 = vid2 + clip + 'detections2.txt ' + view

    args_kalman1 = clip + "detections1.txt" + clip + "trajectories1.txt"
    args_kalman2 = clip + "detections2.txt" + clip + "trajectories2.txt"

    args_traj1 = clip + "detections1.txt" + clip + \
        "trajectories1.txt" + clip + "trajectory1.txt " + view
    args_traj2 = clip + "detections2.txt" + clip + \
        "trajectories2.txt" + clip + "trajectory2.txt " + view

    args_interp1 = clip + 'trajectory1.txt 30' + \
        clip + 'trajectory1.txt ' + view
    args_interp2 = clip + 'trajectory2.txt 30' + \
        clip + 'trajectory2.txt ' + view

    args_reconstruct = session_name.get() + ' ' + clip_name.get() + ' ' + view

    args_topdown = clip + '3d_out.txt ' + clip + 'graphs/top_down.pdf'
    args_sideon = clip + '3d_out.txt ' + clip + 'graphs/side_on.pdf'

    args_trace1 = vid1 + ' ' + clip + \
        'trajectory1.txt ' + clip + 'graphs/trace1.mov'
    args_trace2 = vid2 + ' ' + clip + \
        'trajectory2.txt ' + clip + 'graphs/trace2.mov'

    # New session: create the scene data
    if not os.path.exists(p_session):
        os.makedirs(p_session)
        os.system("./calibrate.py " + args_cal1)
        os.system("./calibrate.py " + args_cal2)
        os.system("./postPoints.py " + args_posts1)
        os.system("./postPoints.py " + args_posts2)
        os.system("./manualMatch.py " + args_match)

    # New clip: create the clip and reconstruction data
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
        os.system("./reconstruct.py " + args_reconstruct)

    # Clip never analysed before: Create the graphs directory
    if not os.path.isdir(p_clip + '/graphs'):
        os.makedirs(p_clip + '/graphs')

    # Everything's there: Visualise it
    if videos:
        os.system("./trace.py " + args_trace1)
        os.system("./trace.py " + args_trace2)
    os.system("./top_down.py " + args_topdown)
    os.system("./side_on.py " + args_sideon)

root = Tk()
root.title("squawkFly")

lst = os.listdir('sessions')

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

session_entry = ttk.Combobox(frame, textvariable=session_name)
session_entry.grid(column=2, row=1, sticky=(W, E))
session_entry['values'] = lst
session_entry.bind('<<ComboboxSelected>>', changeClipOptions)

clip_entry = ttk.Combobox(frame, textvariable=clip_name)
clip_entry.grid(column=2, row=2, sticky=(W, E))

calib1_entry = ttk.Entry(frame, width=60, textvariable=calib1)
clip1_entry = ttk.Entry(frame, width=60, textvariable=clip1)

calib2_entry = ttk.Entry(frame, width=60, textvariable=calib2)
clip2_entry = ttk.Entry(frame, width=60, textvariable=clip2)

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
button_sub = ttk.Button(frame, text="Analyse", command=submit)
button_sub.grid(column=2, row=7, sticky=(W, E))

viewer = StringVar()
button_view = ttk.Checkbutton(frame,
                              text="View process?",
                              variable=viewer,
                              onvalue='view',
                              offvalue='suppress')
button_view.grid(column=3, row=7, sticky=(W, E))
button_view.instate(['disabled'])
viewer.set('suppress')

button_del = ttk.Button(frame, text="Delete current analysis", command=delete)
button_del.grid(column=3, row=8, sticky=(W, E))

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
root.resizable(0, 0)
root.minsize(root.winfo_width(), root.winfo_height())

calib1_entry.focus()
root.bind('<Return>', submit)
root.mainloop()

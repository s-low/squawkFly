#!/usr/local/bin/python

''' plot.py

Plot the detections output by detect.py

Plots can be static and aggregated so that we see all the detections across
the duration of the clip at once, or they can be animated and optionally
'stacked'.

Animate but don't stack = only show the detections in the current frame
Animate and stack = add the detections in the current frame, leave them there

Animation can be saved to file.

Option to supply a different infile and plot that instead.

'''

import sys
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# FLAGS
save = False
animate_on = False
stack = True

# check for stacking and saving
if len(sys.argv) > 1:
    if sys.argv[1] == 'w':
        save = True
    if sys.argv[1] == 's':
        stack = True

# DATA
filename = "data/data_detections.txt"

# Check if a different infile supplied
if len(sys.argv) > 1:
    filename = sys.argv[1]

with open(filename) as datafile:
    data = datafile.read()
    datafile.close()

data = data.split('\n')

all_x = [row.split(' ')[0] for row in data]
all_y = [row.split(' ')[1] for row in data]
all_frames = [row.split(' ')[2] for row in data]

# now translate into frame array
max_frame = int(all_frames[-1])
frame_array = [{} for x in xrange(max_frame + 1)]

# for each data point - dump it into the frame of dictionaries
for i in range(0, max_frame + 1):
    frame_array[i]["x"] = []
    frame_array[i]["y"] = []

# for each recorded frame
for row in data:
    x = row.split(' ')[0]
    y = row.split(' ')[1]
    f = int(row.split(' ')[2])

    frame_array[f]["x"].append(x)
    frame_array[f]["y"].append(y)

# frame_array is an array of dictionaries - each containing the dataset for a
# particular frame. The animation need simply update the dataset with the data
# from that particular frame

# set up the figure, the axis, and the plot element we want to animate
dpi = 113
h = 800 / dpi
w = 1280 / dpi
fig = plt.figure(figsize=(w, h))

ax = plt.axes(xlim=(0, 1280), ylim=(-720, 0))
ax.set_title("Ball Candidate Detections", y=1.03)
ax.set_xlabel("Graphical X")
ax.set_ylabel("Graphical Y")
ax.set_aspect('equal')

if animate_on:
    counter = ax.text(710, -40, 'Frame:', fontsize=15)
scat, = ax.plot([], [], 'k.')

x_set = []
y_set = []


# initialization function: plot the background of each frame
def init():
    scat.set_data([], [])
    return scat,


def animate(i, fig, counter):
    global max_frame
    global x_set
    global y_set

    counter.set_text('Frame: ' + str(i))

    if stack:
        x_set = x_set + frame_array[i]["x"]
        y_set = y_set + frame_array[i]["y"]
    else:
        x_set = frame_array[i]["x"]
        y_set = frame_array[i]["y"]

    scat.set_data(x_set, y_set)

    # re-initialise the datasets to empty for the repeat animation
    if int(i) == int(max_frame - 1):
        x_set = []
        y_set = []
    return scat

# ANIMATE
if animate_on:
    anim = animation.FuncAnimation(fig, animate, fargs=(
        fig, counter), init_func=init, frames=max_frame,
        interval=40, blit=False)
    if save:
        anim.save('../res/animations/test.mp4', fps=30,
                  extra_args=['-vcodec', 'libx264'])
else:
    scat.set_data(all_x, all_y)

# start animation
plt.show()

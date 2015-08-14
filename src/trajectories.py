#!/usr/local/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def pixLength(set_x, set_y):

    length = 0
    if len(set_x) > 3:
        x0 = float(set_x.pop(0))
        y0 = float(set_y.pop(0))
        for x, y in zip(set_x, set_y):
            x = float(x)
            y = float(y)

            dx = abs(x0 - x)
            dy = abs(y0 - y)

            d = ((dx ** 2) + (dy ** 2)) ** 0.5
            length += d
            x0 = x
            y0 = y

    return length


font = {'family': 'normal',
        'weight': 'bold',
        'size': 18}

plt.rc('font', **font)

min_length = 0
if len(sys.argv) > 1:
    min_length = int(sys.argv[1])

# get the trajectory data
with open("data/data_trajectories.txt") as datafile:
    trajectories = datafile.read()
    datafile.close()

# Get the original data points for overlay
with open("data/data_detections.txt") as datafile:
    raw = datafile.read()
    datafile.close()

outfile = open('data_trajectories_subset.txt', 'w')

trajectories = trajectories.split('\n')
raw = raw.split('\n')

# remove the newline at end of trajectory file
trajectories.pop(-1)

# RAW / ORIGINAL
raw_x = [row.split(' ')[0] for row in raw]
raw_y = [row.split(' ')[1] for row in raw]

dpi = 113
h = 800 / dpi
w = 1280 / dpi
fig = plt.figure(figsize=(w, h))

# xlim=(400, 800), ylim=(-720,0)
ax = plt.axes(xlim=(0, 1280), ylim=(-720, 0))
ax.set_title("Ball Trajectory from Kalman Filter", y=1.03)
ax.set_xlabel("Graphical X")
ax.set_ylabel("Graphical Y")
ax.plot(raw_x, raw_y, 'k.')

# TRAJECTORIES
last_tid = int(0)
tid = int(0)

set_x = []
set_y = []
set_f = []

longest = 0
longest_x = []
longest_y = []
longest_f = []

longest_tid = 0
displayed_tids = []

# Rows are: TID, X, Y, FRAME, PID
for row in trajectories:
    tid = int(row.split(' ')[0])
    x = row.split(' ')[1]
    y = row.split(' ')[2]
    f = row.split(' ')[3]

    # add the point to the current trajectory
    if tid == last_tid:
        set_x.append(x)
        set_y.append(y)
        set_f.append(f)

    # we've reached the end of the trajectory (also the  veryfirst point)
    else:

        # check to see if the last T was longest (in pix)
        length = pixLength(set_x, set_y)
        if length > longest:
            longest = length
            longest_x = set_x
            longest_y = set_y
            longest_f = set_f
            longest_tid = last_tid

        # check to see if the last T is above the min selection length
        if len(set_x) >= min_length and min_length != -1:
            displayed_tids.append(last_tid)
            ax.plot(set_x, set_y, linewidth=4)

            # write (x, y, frame) to subset file
            for a, b, c in zip(set_x, set_y, set_f):
                outfile.write(a + ' ' + b + ' ' + c + '\n')

        # reset and start the new T
        set_x = []
        set_y = []
        set_f = []
        last_tid = tid
        set_x.append(x)
        set_y.append(y)
        set_f.append(f)

# file over - handle the remainder T:

# is it longer than the longest T yet?
length = pixLength(set_x, set_y)
if length > longest:
    longest = length
    longest_x = set_x
    longest_y = set_y
    longest_f = set_f
    longest_tid = tid

# is it longer than the minimum length? (exluding -1)
if len(set_x) >= min_length and min_length != -1:
    displayed_tids.append(tid)
    ax.plot(set_x, set_y, linewidth=2)

    # write (x, y, frame) to subset file
    for a, b, c in zip(set_x, set_y, set_f):
        outfile.write(a + ' ' + b + ' ' + c + '\n')

# plot the longest T found if option selected
if min_length == -1:
    ax.plot(longest_x, longest_y, linewidth=4)
    print "Longest trajectory TID:", longest_tid
    print "Length:", round(longest, 1), 'pix'

    # write (x, y, f) to subset file
    for a, b, c in zip(longest_x, longest_y, longest_f):
        outfile.write(a + ' ' + b + ' ' + c + '\n')

else:
    print "Showing trajectories:", displayed_tids

outfile.close()
plt.show()

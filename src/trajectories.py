#!/usr/local/bin/python

''' trajectories.py

    select the longest tip-to-tail trajectory from the candidates outputted
    by Kalman.py

    Write that trajectory data to a new file.

    optionally show all trajectories longer than a minimum length.

    arg1 = minimum length of a trajectory to be included in the subset
        (-1 for longest trajectory)

    arg2 = optional infile for infile of raw detections

    arg3 = optional infile for candidate trajectories

    arg4 = optional outfile for subset of trajectories
'''

import sys
import numpy as np
import matplotlib.pyplot as plt


# default to showing the detection streams
view = True
try:
    if sys.argv[5] == 'suppress':
        view = False
except IndexError:
    pass


# how long is the trajectory from start to finish across the screen
def pixLength(set_x, set_y):

    copy_x = [x for x in set_x]
    copy_y = [y for y in set_y]

    length = 0
    if len(set_x) > 3:
        x0 = float(copy_x.pop(0))
        y0 = float(copy_y.pop(0))
        for x, y in zip(copy_x, copy_y):
            x = float(x)
            y = float(y)

            dx = abs(x0 - x)
            dy = abs(y0 - y)

            d = ((dx ** 2) + (dy ** 2)) ** 0.5
            length += d
            x0 = x
            y0 = y

    return length


min_length = 0
if len(sys.argv) > 1:
    min_length = int(sys.argv[1])

try:
    infile_detections = sys.argv[2]
except IndexError:
    infile_detections = 'data/data_detections.txt'

try:
    infile_trajectories = sys.argv[3]
except IndexError:
    infile_trajectories = 'data/data_trajectories.txt'

try:
    outfilename = sys.argv[4]
except IndexError:
    outfilename = 'data_trajectories_subset.txt'

# get the trajectory data
with open(infile_trajectories) as datafile:
    trajectories = datafile.read()
    datafile.close()

# Get the original data points for overlay
with open(infile_detections) as datafile:
    raw = datafile.read()
    datafile.close()

outfile = open(outfilename, 'w')

trajectories = trajectories.split('\n')
raw = raw.split('\n')

# get rid of any empty line at the end of file
if raw[-1] in ['\n', '\r\n', '']:
    raw.pop(-1)
if trajectories[-1] in ['\n', '\r\n', '']:
    trajectories.pop(-1)

# RAW / ORIGINAL
raw_x = [row.split()[0] for row in raw]
raw_y = [row.split()[1] for row in raw]

dpi = 113
h = 800 / dpi
w = 1280 / dpi
fig = plt.figure(figsize=(w, h))

ax = plt.axes(xlim=(0, 1280), ylim=(-720, 0))
ax.set_title("Ball Trajectory from Kalman Filter", y=1.03)
ax.set_xlabel("Graphical X")
ax.set_ylabel("Graphical Y")
ax.plot(raw_x, raw_y, 'k.')

# TRAJECTORIES
last_tid = int(1)
tid = int(1)

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
    tid = int(row.split()[0])
    x = row.split()[1]
    y = row.split()[2]
    f = row.split()[3]

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
            ax.plot(set_x, set_y, linewidth=2)

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
    ax.plot(longest_x, longest_y, linewidth=2)
    print "Longest trajectory TID:", longest_tid
    print "Pixel Length:", round(longest, 1), 'pix'
    print "Detection Length:", len(longest_x)

    # write (x, y, f) to subset file
    for a, b, c in zip(longest_x, longest_y, longest_f):
        outfile.write(a + ' ' + b + ' ' + c + '\n')

else:
    print "Showing trajectories:", displayed_tids

outfile.close()
if view:
    plt.show()

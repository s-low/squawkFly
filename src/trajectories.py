#!/usr/local/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

min_length = 0
if len(sys.argv) > 1:
    min_length = int(sys.argv[1])

# DATA
with open("data/data_trajectories.txt") as datafile:
    trajectories = datafile.read()
    datafile.close()

with open("data/data_detections.txt") as datafile:
    raw = datafile.read()
    datafile.close()

outfile = open('data_trajectories_subset.txt', 'w')

trajectories = trajectories.split('\n')
raw = raw.split('\n')

# remove the newline at EOF
trajectories.pop(-1)

raw_x = [row.split(' ')[0] for row in raw]
raw_y = [row.split(' ')[1] for row in raw]

dpi = 113
h = 800 / dpi
w = 1280 / dpi
fig = plt.figure(figsize=(w, h))

# xlim=(400, 800), ylim=(-720,0)
ax = plt.axes(xlim=(0, 1280), ylim=(-720, 0))
ax.set_title("Points from trajectories Filter", y=1.03)
ax.set_xlabel("Graphical X")
ax.set_ylabel("Graphical Y")

ax.plot(raw_x, raw_y, 'k.')
last_t = int(0)
set_x = []
set_y = []
set_f = []
current_longest_x = []
current_longest_y = []
current_longest_t = 0
displayed_t = []

# Rows are: TID, X, Y, FRAME, PID
for row in trajectories:
    t = int(row.split(' ')[0])
    x = row.split(' ')[1]
    y = row.split(' ')[2]
    f = row.split(' ')[3]

    # build the trajectory
    if t == last_t:
        set_x.append(x)
        set_y.append(y)
        set_f.append(f)

    # work out what to do with it
    else:

        # log the longest T
        if len(set_x) > len(current_longest_x):
            current_longest_x = set_x
            current_longest_y = set_y
            current_longest_f = set_f
            current_longest_t = t

        # accepting anything over a certain length
        if len(set_x) >= min_length and min_length != -1:
            displayed_t.append(t)
            ax.plot(set_x, set_y, linewidth=2)

            # write (x, y, frame) to subset file
            for a, b, c in zip(set_x, set_y, set_f):
                outfile.write(a + ' ' + b + ' ' + c + '\n')

        # reset and start the new T
        set_x = []
        set_y = []
        last_t = t
        set_x.append(x)
        set_y.append(y)
        set_f.append(f)

# file over - handle the remainder T
if len(set_x) > len(current_longest_x):
    current_longest_x = set_x
    current_longest_y = set_y
    current_longest_f = set_f
    current_longest_t = t

if len(set_x) >= min_length and min_length != -1:
    ax.plot(set_x, set_y, linewidth=2)

if min_length == -1:
    ax.plot(current_longest_x, current_longest_y, linewidth=2)
    print "Longest trajectory TID:", current_longest_t
    print "Length:", len(current_longest_x)

    # write (x, y, frame) to subset file
    for a, b, c in zip(current_longest_x, current_longest_y, current_longest_f):
        outfile.write(a + ' ' + b + ' ' + c + '\n')

else:
    print "Showing trajectories:", displayed_t

outfile.close()
plt.show()

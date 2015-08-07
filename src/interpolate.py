#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import os.path

plt.style.use('ggplot')

max_ms_diff = float(20)  # no two points should be further apart in time

try:
    filename = sys.argv[1]
    frame_rate = float(sys.argv[2])
except IndexError:
    print "Usage: ./interpolate <file> <framerate>"
    sys.exit()

if abs(frame_rate - 24) < 0.01:
    frame_rate = 23.976

name, ext = os.path.splitext(filename)

outfilename = name + "_interpolated.txt"
frame_length_ms = float(1000) / float(frame_rate)

print "Frame Rate:", frame_rate
print "Length of each frame (ms):", frame_length_ms

# data in format: x / y / frame / pid
with open(filename) as datafile:
    data = datafile.read()
    datafile.close()

data = data.split('\n')
points = []
interpolated_points = []

# rows are X, Y, FRAME
for row in data:
    point = []
    point.append(float(row.split(' ')[0]))
    point.append(float(row.split(' ')[1]))
    point.append(int(row.split(' ')[2]))
    points.append(point)


def fit(a, b):

    z = np.polyfit(a, b, 4)
    func = np.poly1d(z)

    a_new = np.linspace(a[0], a[-1], 50)
    b_new = func(a_new)

    plt.plot(a, b, 'o', a_new, b_new)
    plt.xlim([a[0] - 1, a[-1] + 1])

    plt.show()

    return func


def interpolate(start, end, func):
    global interpolated_points
    print "INTERPOLATE between:", start, end

    for i in range(start, end):
        p = points[i]  # this point
        n = points[i + 1]  # next point

        print "\np: ", p
        print "n: ", n, "\n"

        x = p[0]
        y = p[1]

        frames_between = n[2] - p[2]
        ms_between = frames_between * frame_length_ms
        num_between = int(ms_between / max_ms_diff)

        diff_x = n[0] - p[0]

        dx = float(diff_x / (num_between + 1))

        interpolated_points.append(p)

        if num_between != 0:
            print "> adding points:"
        for j in range(0, num_between):
            new_point = [None] * 3
            new_point[0] = x + dx
            new_point[1] = func(x + dx)
            new_point[2] = p[2]
            x = x + dx
            interpolated_points.append(new_point)
            print new_point

    # plot the current interpolated set of points with f overlain
    arr = np.array(interpolated_points)
    x = arr[:, 0]
    y = arr[:, 1]

    x_new = np.linspace(seg_x[0], seg_x[-1], 50)
    y_new = func(x_new)

    plt.plot(x, y, 'o', x_new, y_new)

    # plt.xlim([seg_x[0] - 1, seg_x[-1] + 1])

    plt.show()


# Find any bounces in the trajectory
arr = np.array(points)
x = arr[:, 0]
y = arr[:, 1]

bounc_i = []
for i, ay in enumerate(y):

    if i == 0:
        prev = -1000
    else:
        prev = y[i - 1]

    try:
        nex = y[i + 1]
    except IndexError:
        nex = -1000

    if ay < prev and ay < nex:
        ax = x[i]
        print "\nBounce at:", ax, ay
        print "Previous, next y:", prev, nex
        print "Index:", i
        bounc_i.append(i)

# Split at the bounces and interpolate each individually
root = 0

# for each bounce
for count, i in enumerate(bounc_i):

    print "\n---New Segment---"
    # get the segment
    seg_x = x[root:i + 1]
    seg_y = y[root:i + 1]
    print "Start:", root
    print "End:", i

    # fit it
    f = fit(seg_x, seg_y)

    # interpolate the original data between along the segment
    interpolate(root, i, f)

    root = i

# corner case: no bounces / last bounce
print "\n---New Segment---"
seg_x = x[root:]
seg_y = y[root:]
print "Start:", root
print "End:", len(x)
f = fit(seg_x, seg_y)
interpolate(root, len(x) - 1, f)

# write to file
outfile = open(outfilename, 'w')
startOfFile = True

for p in interpolated_points:
    if not startOfFile:
        outfile.write('\n')

    p_string = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + '1'
    outfile.write(p_string)
    startOfFile = False

outfile.close()
print "> Written to:", outfilename

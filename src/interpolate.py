#!/usr/local/bin/python

''' interpoate.py

Interpolate an extracted trajectory.

trajectory is first split at bounce points into complete segments, which
are fit with 5th degree polynomial and interpolated.

arg1 = input list of trajectory image points
arg2 = frame rate, float.
*arg3* = optional outfilename
*arg4* = optional 'suppress' of graphics

'''

import sys
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
import os.path

try:
    filename = sys.argv[1]
    frame_rate = float(sys.argv[2])
except IndexError:
    print "Usage: ./interpolate <file> <framerate>"
    sys.exit()

try:
    outfilename = sys.argv[3]
except IndexError:
    name, ext = os.path.splitext(filename)
    outfilename = name + "_interpolated.txt"

# default to showing the detection streams
view = True
try:
    if sys.argv[4] == 'suppress':
        view = False
except IndexError:
    pass

# if the user thinks the camera is 24fps, correct it slightly
if abs(frame_rate - 24) < 0.01:
    frame_rate = 23.976

frame_length_ms = float(1000) / float(frame_rate)

# no two points should be further apart in time than 17ms after interpolation
max_ms_diff = float(frame_length_ms / 2)

print "Frame Rate:", frame_rate
print "Length of each frame (ms):", frame_length_ms

# data in format: x / y / frame / pid
with open(filename) as datafile:
    data = datafile.read()
    datafile.close()
data = data.split('\n')

# Gobble blanks at EOF if there
if data[-1] in ['\n', '\r\n', '']:
    data.pop(-1)

points = []
interpolated_points = []

# rows are X, Y, FRAME
for row in data:
    point = []
    point.append(float(row.split(' ')[0]))
    point.append(float(row.split(' ')[1]))
    point.append(int(row.split(' ')[2]))
    points.append(point)


# fit the curve segment with 5th degree polynomial
def fit(a, b):

    z = np.polyfit(a, b, 5)
    func = np.poly1d(z)

    a_new = np.linspace(a[0], a[-1], 50)
    b_new = func(a_new)

    if view:
        plt.plot(a, b, 'o', a_new, b_new)
        plt.xlim([a[0] - 1, a[-1] + 1])
        plt.show()

    return func


# fill in the gaps, func is the fit function
def interpolate(start, end, func):
    global interpolated_points

    for i in range(start, end):
        p = points[i]  # this point
        n = points[i + 1]  # next point

        x = p[0]
        y = p[1]

        frames_between = n[2] - p[2]
        ms_between = frames_between * frame_length_ms
        num_between = int(ms_between / max_ms_diff)

        diff_x = n[0] - p[0]

        dx = float(diff_x / (num_between + 1))

        interpolated_points.append(p)

        for j in range(0, num_between):
            new_point = [None] * 3
            new_point[0] = x + dx
            new_point[1] = func(x + dx)
            new_point[2] = p[2]
            x = x + dx
            interpolated_points.append(new_point)

    # plot the current interpolated set of points with f overlain
    arr = np.array(interpolated_points)
    x = arr[:, 0]
    y = arr[:, 1]

    x_new = np.linspace(seg_x[0], seg_x[-1], 50)
    y_new = func(x_new)
    if view:
        plt.plot(x, y, 'o', x_new, y_new)
        plt.show()


# FIRST: Find any bounces in the trajectory
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
        bounc_i.append(i)

# Split trajectory at the bounces and interpolate each segment individually
root = 0

# for each bounce
for count, i in enumerate(bounc_i):

    # get the segment
    seg_x = x[root:i + 1]
    seg_y = y[root:i + 1]

    # fit it
    f = fit(seg_x, seg_y)

    # interpolate the original data between along the segment
    interpolate(root, i, f)

    root = i

# corner case: no bounces / last bounce
seg_x = x[root:]
seg_y = y[root:]

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

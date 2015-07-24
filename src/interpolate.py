#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

try:
    filename = sys.argv[1]
    frame_rate = int(sys.argv[2])
except IndexError:
    print "Usage: ./interpolate <file> <framerate>"
    sys.exit()

outfilename = "interpolation/data_out.txt"
frame_length_ms = int(1000 / frame_rate)


# data in format: x / y / frame / pid
with open(filename) as datafile:
    data = datafile.read()
    datafile.close()

data = data.split('\n')
points = []
interpolated_points = []

for row in data:
    point = []
    point.append(float(row.split(' ')[0]))
    point.append(float(row.split(' ')[1]))
    point.append(int(row.split(' ')[2]))
    points.append(point)

# fit the data with a polynomial f
arr = np.array(points)
x = arr[:, 0]
y = arr[:, 1]
z = np.polyfit(x, y, 4)
f = np.poly1d(z)

x_new = np.linspace(x[0], x[-1], 50)
y_new = f(x_new)

plt.plot(x, y, 'o', x_new, y_new)
plt.xlim([x[0] - 1, x[-1] + 1])

plt.show()

for i in range(0, len(points) - 1):
    p = points[i]  # this point
    n = points[i + 1]  # next point

    print "\np: ", p
    print "n: ", n, "\n"

    x = p[0]
    y = p[1]

    max_ms_diff = 20  # no two points should be further apart than this

    frames_between = n[2] - p[2]
    ms_between = frames_between * frame_length_ms
    num_between = int(ms_between / max_ms_diff)

    diff_x = n[0] - p[0]
    # diff_y = n[1] - p[1]

    dx = int(diff_x / (num_between + 1))
    # dy = int(diff_y / (num_between + 1))

    interpolated_points.append(p)

    print "> adding points:"
    for j in range(0, num_between):
        new_point = [None] * 3
        new_point[0] = x + dx
        new_point[1] = f(x + dx)
        new_point[2] = p[2]
        x = x + dx
        interpolated_points.append(new_point)
        print new_point

    outfile = open(outfilename, 'w')
    startOfFile = True

    for p in interpolated_points:
        if not startOfFile:
            outfile.write('\n')

        p_string = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + '1'
        outfile.write(p_string)
        startOfFile = False

    outfile.close()

arr = np.array(interpolated_points)
x = arr[:, 0]
y = arr[:, 1]
plt.plot(x, y, 'o', x_new, y_new)
plt.xlim([x[0] - 1, x[-1] + 1])

plt.show()

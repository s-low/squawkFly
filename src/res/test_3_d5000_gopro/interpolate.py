#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np

frame_length_ms = 33

with open('data_sparse.txt') as datafile:
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

for i in range(0, len(points) - 1):
    p = points[i]
    n = points[i + 1]

    print "\np: ", p
    print "n: ", n, "\n"

    x = p[0]
    y = p[1]

    frames_between = n[2] - p[2]
    ms_between = frames_between * frame_length_ms
    num_between = int(ms_between / 10)

    diff_x = n[0] - p[0]
    diff_y = n[1] - p[1]

    dx = int(diff_x / (num_between + 1))
    dy = int(diff_y / (num_between + 1))

    interpolated_points.append(p)

    print "> adding points:"
    for j in range(0, num_between):
        new_point = [None] * 3
        new_point[0] = x + dx
        new_point[1] = y + dy
        new_point[2] = p[2]
        x = x + dx
        y = y + dy
        interpolated_points.append(new_point)
        print new_point

    outfile = open('data_interpolated.txt', 'w')

    for p in interpolated_points:
        p_string = str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + ' ' + '1\n'
        outfile.write(p_string)

    outfile.close()

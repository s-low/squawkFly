#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np

frame_length_ms = 33

with open('data_interpolate.txt') as datafile:
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

    frames_between = n[2] - p[2]
    ms_between = frames_between * frame_length_ms
    num_between = int(ms_between / 10)

    diff_x = n[0] - p[0]
    diff_y = n[1] - p[1]

    dx = int(diff_x / num_between)
    dy = int(diff_y / num_between)

    for j in range(0, num_between):
        new_point = [None] * 3

#!/usr/local/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt

# DATA
with open("kalman_points.txt") as datafile:
	data = datafile.read()
	datafile.close()

data = data.split('\n')

all_x = [row.split(' ')[0] for row in data]
all_y = [row.split(' ')[1] for row in data]

dpi = 113
h = 800 / dpi
w = 1280 / dpi
fig = plt.figure(figsize=(w,h))

ax = plt.axes(xlim=(0, 1280), ylim=(0, 720))
ax.set_title("Points from Kalman Filter", y = 1.03)
ax.set_xlabel("Graphical X")
ax.set_ylabel("Graphical Y")

scat, = ax.plot([], [], 'ro')

scat.set_data(all_x,all_y)

plt.show()

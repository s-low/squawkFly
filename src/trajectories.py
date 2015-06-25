#!/usr/local/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# DATA
with open("kalman_points.txt") as datafile:
	kalman = datafile.read()
	datafile.close()

with open("output.txt") as datafile:
	raw = datafile.read()
	datafile.close()

kalman = kalman.split('\n')
raw = raw.split('\n')

# remove the newline at EOF
kalman.pop(-1)


raw_x = [row.split(' ')[0] for row in raw]
raw_y = [row.split(' ')[1] for row in raw]

dpi = 113
h = 800 / dpi
w = 1280 / dpi
fig = plt.figure(figsize=(w,h))

# xlim=(400, 800), ylim=(-720,0)
ax = plt.axes()
ax.set_title("Points from Kalman Filter", y = 1.03)
ax.set_xlabel("Graphical X")
ax.set_ylabel("Graphical Y")

# scat, = ax.plot([], [], 'ro')
# first_row = kalman[0]
# first_t = first_row.split(' ')[0]

# print first_t
last_t = int(0)
set_x = []
set_y = []

for row in kalman:
	t = int(row.split(' ')[0])
	x = row.split(' ')[1]
	y = row.split(' ')[2]

	if t == last_t:
		set_x.append(x)
		set_y.append(y)

	else:
		ax.plot(set_x, set_y)
		set_x = []
		set_y = []
		last_t = t
		set_x.append(x)
		set_y.append(y)

ax.plot(set_x, set_y)
ax.plot(raw_x, raw_y, 'ro')

# scat.set_kalman(all_x,all_y)

plt.show()

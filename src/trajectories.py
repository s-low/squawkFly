#!/usr/local/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

min_length = 0
if len(sys.argv) > 1:
	min_length = int(sys.argv[1])

# DATA
with open("data_trajectories.txt") as datafile:
	trajectories = datafile.read()
	datafile.close()

with open("data_detections.txt") as datafile:
	raw = datafile.read()
	datafile.close()

trajectories = trajectories.split('\n')
raw = raw.split('\n')

# remove the newline at EOF
trajectories.pop(-1)

raw_x = [row.split(' ')[0] for row in raw]
raw_y = [row.split(' ')[1] for row in raw]

dpi = 113
h = 800 / dpi
w = 1280 / dpi
fig = plt.figure(figsize=(w,h))

# xlim=(400, 800), ylim=(-720,0)
ax = plt.axes(xlim=(0, 1280), ylim=(-720,0))
ax.set_title("Points from trajectories Filter", y = 1.03)
ax.set_xlabel("Graphical X")
ax.set_ylabel("Graphical Y")

ax.plot(raw_x, raw_y, 'k.')
last_t = int(0)
set_x = []
set_y = []

for row in trajectories:
	t = int(row.split(' ')[0])
	x = row.split(' ')[1]
	y = row.split(' ')[2]

	if t == last_t:
		set_x.append(x)
		set_y.append(y)

	else:
		if len(set_x) > min_length:
			ax.plot(set_x, set_y,linewidth=2)
		set_x = []
		set_y = []
		last_t = t
		set_x.append(x)
		set_y.append(y)

if len(set_x) > min_length:
	ax.plot(set_x, set_y,linewidth=2)


# scat.set_trajectories(all_x,all_y)

plt.show()

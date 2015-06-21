#!/usr/local/bin/python
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# DATA
with open("output.txt") as datafile:
	data = datafile.read()
	datafile.close()

data = data.split('\n')

all_x = [row.split(' ')[0] for row in data]
all_y = [row.split(' ')[1] for row in data]
all_frames = [row.split(' ')[2] for row in data]

# now translate into frame array
max_frame = int(all_frames[-1])

dpi = 113
h = 800 / dpi
w = 1280 / dpi
fig = plt.figure(figsize=(w,h))

ax = plt.axes(xlim=(0, 70), ylim=(-720, 0))
ax.set_title("X position of Ball Candidates by Frame", y = 1.03)
ax.set_xlabel("Frame Number")
ax.set_ylabel("Graphical Y")

scat, = ax.plot([], [], 'ro')

scat.set_data(all_frames, all_y)

plt.show()
#!/usr/local/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# DATA
with open("kalman_points.txt") as datafile:
	data = datafile.read()
	datafile.close()

data = data.split('\n')

# remove the newline at EOF
data.pop(-1)


# all_x = [row.split(' ')[1] for row in data]
# all_y = [row.split(' ')[2] for row in data]

dpi = 113
h = 800 / dpi
w = 1280 / dpi
fig = plt.figure(figsize=(w,h))

ax = plt.axes(xlim=(0, 1280), ylim=(-720,0))
ax.set_title("Points from Kalman Filter", y = 1.03)
ax.set_xlabel("Graphical X")
ax.set_ylabel("Graphical Y")

# scat, = ax.plot([], [], 'ro')
first_row = data[0]
first_t = first_row.split(' ')[0]

last_t = first_t
set_x = []
set_y = []

for row in data:
	t = row.split(' ')[0]
	x = row.split(' ')[1]
	y = row.split(' ')[2]

	if t == last_t:
		print "add to",t
		set_x.append(x)
		set_y.append(y)

	else:
		print "new trajectory"
		ax.plot(set_x, set_y)
		set_x = []
		set_y = []
		last_t = t

ax.plot(set_x,set_y)

# scat.set_data(all_x,all_y)

plt.show()

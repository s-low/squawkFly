#!/usr/bin/python

from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open("output.txt") as datafile:
    data = datafile.read()
    datafile.close()

data = data.split('\n')

# x = [row.split(' ')[0] for row in data]
# y = [row.split(' ')[1] for row in data]
all_frames = [row.split(' ')[2] for row in data]

# now translate into frame array
max_frame = int(all_frames[-1])
frame_array = [{} for x in xrange(max_frame+1)]

# for each data point - dump it into the frame of dictionaries
for i in range(0, max_frame+1):
	frame_array[i]["x"] = []
	frame_array[i]["y"] = []

# for each recorded frame
for row in data:
	x = row.split(' ')[0]
	y = row.split(' ')[1]
	f = int(row.split(' ')[2])
	print 
	frame_array[f]["x"].append(x)
	frame_array[f]["y"].append(y)

print frame_array

# set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-10, 10))
scat, = ax.plot([], [], 'bo')

# initialization function: plot the background of each frame
def init():
    scat.set_data([], [])
    return scat,

def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    scat.set_data(x, y)
    return scat,






# ax = fig.add_subplot(111)
# ax.set_xlabel('Graphical x')
# ax.set_ylabel('Graphical y')
# ax.set_ylim([-500,0])
# ax.set_xlim([0,900])
		
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=False)

# start animation
plt.show()
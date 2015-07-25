#!/usr/local/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open("data/data_sim.txt") as datafile:
    data = datafile.read()
    datafile.close()

data = data.split('\n')

all_x = [float(row.split(' ')[0]) for row in data]
all_y = [float(row.split(' ')[1]) for row in data]
all_z = [float(row.split(' ')[2]) for row in data]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(all_x, all_y, all_z, zdir='z')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

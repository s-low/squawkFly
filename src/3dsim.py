#!/usr/local/bin/python

''' 3dsim.py
Given a text file containing 3 columns of space-delimited coordinates,
plot them as X, Y, Z and put to screen.
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

filename = sys.argv[1]

with open(filename) as datafile:
    data = datafile.read()
    datafile.close()

data = data.split('\n')

# if there is a blank line at EOF - pop it
if data[-1] in ['\n', '\r\n', '']:
    data.pop(-1)

X = [float(row.split()[0]) for row in data]
Y = [float(row.split()[1]) for row in data]
Z = [float(row.split()[2]) for row in data]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, Z, zdir='z')

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array(
    [max(X) - min(X), max(Y) - min(Y), max(Z) - min(Z)]).max()
Xb = 0.5 * max_range * \
    np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (max(X) + min(X))
Yb = 0.5 * max_range * \
    np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (max(Y) + min(Y))
Zb = 0.5 * max_range * \
    np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (max(Z) + min(Z))

# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

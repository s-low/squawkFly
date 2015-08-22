#!/usr/local/bin/python

'''
Open a 3d trajectory file (x y z) and produce a top-down plot of the
y-z plane.
'''

import sys
import matplotlib.pyplot as plt

filename = sys.argv[1]

with open(filename) as datafile:
    data = datafile.read()
    datafile.close()

data = data.split('\n')

# get rid of any empty line at the end of file
if data[-1] in ['\n', '\r\n', '']:
    print "pop"
    data.pop(-1)

x = [row.split()[0] for row in data]
y = [row.split()[1] for row in data]
z = [row.split()[2] for row in data]

y0 = y[4]
z0 = z[4]

y1 = y[-1]
z1 = z[-1]

fig = plt.figure('Top Down Projection')
ax = fig.add_subplot(111, aspect='equal')

ax.plot([z0, z1], [y0, y1])

ax.set_xlabel("Z / m")
ax.set_ylabel("Y / m")
ax.plot(z, y, 'k.')


plt.show()

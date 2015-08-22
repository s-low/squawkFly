#!/usr/local/bin/python

'''
Open a 3d trajectory file (x y z) and produce a top-down plot of the
y-z plane.
'''

import sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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

bly = y[0]
blz = z[0]
tly = y[1]
tlz = z[1]
try_ = y[2]
trz = z[2]
bry = y[3]
brz = z[3]

fig = plt.figure('Top Down Projection')
ax = fig.add_subplot(111, aspect='equal')

ax.set_xlabel("Z / m")
ax.set_ylabel("Y / m")
ax.plot(z, y, 'k.')
ax.plot([tlz, trz], [tly, try_], c='k')
ax.plot([blz, tlz], [bly, tly], c='k')
ax.plot([brz, trz], [bry, try_], c='k')

# 2.14m wall at 9.14m
ax.plot([9.14, 9.14], [0, 1.82], c='r', linewidth=2)

plt.show()

#!/usr/local/bin/python

''' side_on.py

    Open a 3d trajectory file (x y z) and produce a side-on plot of the
    y-z plane, with straight line between start and end and a virtual
    wall superimposed at 10 yards.

    arg1 = infile
    arg2 = optional outfile
'''

import sys
import matplotlib.pyplot as plt

infilename = sys.argv[1]
outfilename = None
try:
    outfilename = sys.argv[2]
except IndexError:
    pass

with open(infilename) as datafile:
    data = datafile.read()
    datafile.close()

data = data.split('\n')

# get rid of any empty line at the end of file
if data[-1] in ['\n', '\r\n', '']:
    data.pop(-1)

x = [row.split()[0] for row in data]
y = [row.split()[1] for row in data]
z = [row.split()[2] for row in data]

# Get goalpost corner points
bly = y.pop(0)
blz = z.pop(0)

tly = y.pop(0)
tlz = z.pop(0)

try_ = y.pop(0)
trz = z.pop(0)

bry = y.pop(0)
brz = z.pop(0)

# Work out everage depth of goalposts
avgz = (float(blz) + float(tlz) + float(trz) + float(brz)) / 4

fig = plt.figure('Side On Projection with Virtual Wall')
ax = fig.add_subplot(111, aspect='equal')

ax.set_xlabel("Distance Travelled to Goal / m")
ax.set_ylabel("Height / m")
ax.plot(z, y, 'k.')

# Draw a red 2.14m wall at 9.14m
ax.plot([9.14, 9.14], [0, 1.82], c='r', linewidth=2)
ax.plot([avgz, avgz], [0, 2.44], c='k', linewidth=2)

plt.show()

# Save it if necessary
if outfilename is not None:
    print "Save:", outfilename
    fig.savefig(outfilename)

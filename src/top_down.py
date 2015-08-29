#!/usr/local/bin/python

'''
Open a 3d trajectory file (x y z) and produce a top-down plot of the
x-z plane.
'''

import sys
import math
import matplotlib.pyplot as plt

left = 'left'
right = 'right'
neither = ''

# return the distance from a cartesian point to a straight line through
# the origin: y = mx + 0


def distanceToLine(m1, point):
    curve = neither
    x = point[0]
    y = point[1]

    # y1 = m1x + k
    if m1 < 0.001:
        m2 = 99990
    else:
        m2 = - 1 / m1

    k = y - (m2 * x)

    intersect_x = float(k) / (float(m1) - float(m2))
    intersect_y1 = (m1 * intersect_x)
    intersect_y2 = (m2 * intersect_x) + k
    assert(abs(intersect_y1 - intersect_y2) < 5), "Line intersection problem."

    d = math.hypot(x - intersect_x, y - intersect_y1)

    if x < intersect_x and y > intersect_y1:
        curve = right
    elif x > intersect_x and y < intersect_y1:
        curve = left
    else:
        curve = neither

    return d, curve

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

# goalpost corners
blx = x[0]
blz = z[0]
tlx = x[1]
tlz = z[1]
trx = x[2]
trz = z[2]
brx = x[3]
brz = z[3]

# start and end points of trajectory for straight line path
x0 = x[4]
z0 = z[4]
x1 = x[-1]
z1 = z[-1]
gradient = (float(z1) - float(z0)) / (float(x1) - float(x0))
print "gradient:", gradient

curve = ''
max_d_left = 0
max_d_right = 0
max_d = 0
for i in xrange(4, len(x)):
    p_x = float(x[i])
    p_z = float(z[i])
    d, c = distanceToLine(gradient, (p_x, p_z))
    if d > max_d_left and c == left:
        max_d_left = d
    elif d > max_d_right and c == right:
        max_d_right = d

# if curved both left and right
if max_d_right > 0 and max_d_left > 0:
    curve = "bidirectional movement."
    max_d = max_d_left + max_d_right

elif max_d_left > 0:
    curve = "left-hand curve."
    max_d = max_d_left

elif max_d_right > 0:
    curve = "right-hand curve."
    max_d = max_d_right

string = str(round(max_d, 2)) + "m of " + curve

fig = plt.figure('Top Down Projection')
ax = fig.add_subplot(111, aspect='equal')

# annotate curve
trans = ax.get_xaxis_transform()
ann = ax.annotate(string, xy=(8, 0.3), xycoords=trans)

# Add the straight line path
ax.plot([x0, x1], [z0, z1])

# add the goalposts
ax.plot([tlx, trx], [tlz, trz], c='k')
ax.plot([blx, tlx], [blz, tlz], c='k')
ax.plot([brx, trx], [brz, trz], c='k')

ax.set_xlabel("Lateral Movement / m")
ax.set_ylabel("Distance Travelled to Goal / m")
ax.set_ylim([0, 25])
ax.plot(x, z, 'k.')

plt.show()
if outfilename is not None:
    print "Save:", outfilename
    fig.savefig(outfilename, bbox_inches='tight')

#!/usr/local/bin/python

'''
Open a 3d trajectory file (x y z) and produce a top-down plot of the
x-z plane.
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
    print "pop"
    data.pop(-1)

x = [row.split()[0] for row in data]
y = [row.split()[1] for row in data]
z = [row.split()[2] for row in data]


blx = x[0]
blz = z[0]
tlx = x[1]
tlz = z[1]
trx = x[2]
trz = z[2]
brx = x[3]
brz = z[3]


x0 = x[4]
z0 = z[4]

x1 = x[-1]
z1 = z[-1]

fig = plt.figure('Top Down Projection')
ax = fig.add_subplot(111, aspect='equal')

print x0, z0
print x1, z1

ax.plot([x0, x1], [z0, z1])
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
    fig.savefig(outfilename)

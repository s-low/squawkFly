#!/usr/local/bin/python

''' ground_truth_separations.py

Simple script to calculate the average 3-space distance between a set of X Y Z
ground truth image points and the two camera centres used in simulation.

This is for the analysis of recontruction error with increasing depth.

'''


import os.path
import sys
import math


# distance between two 3d coordinates
def sep3D(a, b):
    xa = a[0]
    ya = a[1]
    za = a[2]

    xb = b[0]
    yb = b[1]
    zb = b[2]

    dist = math.sqrt(((xa - xb) ** 2) + ((ya - yb) ** 2) + ((za - zb) ** 2))

    return dist

# Supply
datafile = open(sys.argv[1], 'r')
all_data = datafile.read()
datafile.close()

all_data = all_data.split('\n')
if all_data[-1] in ['\n', '\r\n', '']:
    all_data.pop(-1)

all_data.pop(0)
all_data.pop(0)
all_data.pop(0)
all_data.pop(0)

# Hardcoded camera centres used in the simulation
c1 = (-5, 1, -14)
c2 = (5, 1, -13)

folder = os.path.dirname(sys.argv[1])
outfile = open(os.path.join(folder, 'ground_truth_point_distances.txt'), 'w')

for p in all_data:
    x = float(p.split()[0])
    y = float(p.split()[1])
    z = float(p.split()[2])
    d1 = sep3D((x, y, z), c1)
    d2 = sep3D((x, y, z), c2)
    d = (d1 + d2) / 2
    outfile.write(str(d) + '\n')

outfile.close()

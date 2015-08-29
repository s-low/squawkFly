#!/usr/local/bin/python

import sys
import os.path
import numpy as np


def sep(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])

    d = ((dx ** 2) + (dy ** 2)) ** 0.5

    return d


folder = sys.argv[1]

f_detections = os.path.join(folder, 'detections.txt')
f_trajectory = os.path.join(folder, 'trajectory.txt')
f_truth = os.path.join(folder, 'ground_truth.txt')
f_out = os.path.join(folder, 'analysis.txt')

print "Detections:", f_detections
print "Trajectory:", f_trajectory
print "Truth:", f_truth
print "Out:", f_out


with open(f_detections) as datafile:
    detections = datafile.read()
    datafile.close()

with open(f_trajectory) as datafile:
    trajectory = datafile.read()
    datafile.close()

with open(f_truth) as datafile:
    truth = datafile.read()
    datafile.close()

trajectory = trajectory.split('\n')
detections = detections.split('\n')
truth = truth.split('\n')

if trajectory[-1] in ['\n', '\r\n', '']:
    trajectory.pop(-1)

if truth[-1] in ['\n', '\r\n', '']:
    truth.pop(-1)

if detections[-1] in ['\n', '\r\n', '']:
    detections.pop(-1)

num_detections = len(detections)
num_truth = len(truth)
num_trajectory = len(trajectory)

print "Detections:", num_detections
print "Truths:", num_truth
print "Trajectory:", num_trajectory

separations = []
tracked = 0
detected = 0

for g in truth:
    g_x = g.split()[0]
    g_y = g.split()[1]
    g_f = g.split()[2]

    for d in detections:
        d_x = d.split()[0]
        d_y = d.split()[1]
        d_f = d.split()[2]

        if d_f == g_f:
            dist = sep((g_x, g_y), (d_x, d_y))
            if dist < 25:
                # truth has been detected
                detected += 1
                separations.append(dist)

                # is the detected truth in the trajectory
                for t in trajectory:
                    t_x = t.split()[0]
                    t_y = t.split()[1]
                    t_f = t.split()[2]
                    if t_f == g_f:
                        if t_x == d_x and t_y == d_y:
                            tracked += 1


mean = np.mean(separations)
std = np.std(separations)

outfile = open(f_out, 'w')
outfile.write("num_detections:" + str(num_detections))
outfile.write("\nnum_truth:" + str(num_truth))
outfile.write("\nnum_detected:" + str(detected))
outfile.write("\nmean sep:" + str(mean))
outfile.write("\nstd sep:" + str(std))
outfile.write("\nnum in trajectory:" + str(num_trajectory))
outfile.write("\nnum of truths in trajectory:" + str(tracked))
outfile.close()

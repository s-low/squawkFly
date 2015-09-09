#!/usr/local/bin/python

''' stitch.py

    Now unused post processing step that appends overlapping trajectories to
    one another. Was extremely important when the Kalman filter didn't work.
'''

import sys
import numpy as np
import math

with open("data/data_trajectories.txt") as datafile:
    data = datafile.read()
    datafile.close()

data = data.split("\n")
data.pop(-1)

# for each pair of trajectories A and B in data
# each point will be (tid, x, y, frame, pid)
tid_pid = []

# index markers
TID = 0
XC = 1
YC = 2
FRAME = 3
PID = 4

# debug mode
d = False

for row in data:
    tid_pid.append(list((row.split(' ')[0],
                         row.split(' ')[1],
                         row.split(' ')[2],
                         row.split(' ')[3],
                         row.split(' ')[4])))

tid_list = []

for point in tid_pid:
    if point[TID] not in tid_list:
        tid_list.append(point[TID])

int_list = map(int, tid_list)
max_tid = max(int_list)


def stitch():
    # modify scope
    global tid_list
    global tid_pid

    changed = False
    matches = 0
    # for each trajectory
    for A in tid_list:

        # for each other trajectory
        for B in tid_list:
            if A != B:

                # compile and RECOMPILE the A points into a list
                A_points = []
                for point in tid_pid:
                    if point[TID] == A:
                        A_points.append(point[PID])

                # compile another list of points
                B_points = []
                for point in tid_pid:
                    if point[TID] == B:
                        B_points.append(point[PID])

                if len(B_points) > 1 and len(A_points) > 1:

                    # if A ends somewhere in B
                    if A_points[-1] in B_points:

                        # assume they match to begin
                        match = True

                        # how many places into B does the end of A sit
                        offset = B_points.index(A_points[-1])

                        if offset == 0:
                            # compare the angle between end of A and start of B
                            theta1 = getAngle(A_points[-2], A_points[-1])
                            theta2 = getAngle(B_points[0], B_points[1])

                            # if the angles are within k degrees
                            if abs(theta1 - theta2) < 10:
                                changed = True
                                if d:
                                    print "\n PARTIAL MATCH between TIDS:", A, B
                                    print "PID:", A_points[-1]
                                    raw_input()
                                del B_points[0:1]

                                # do this to original data
                                for point in tid_pid:
                                    if point[TID] == B and point[PID] in B_points:
                                        point[TID] = A
                                    elif point[TID] == B and point[PID] \
                                            not in B_points:
                                        point[0] = 1000

                        elif offset > 0 and offset < len(A_points):
                            # check every element of B from B[0] to B[offset]
                            for i in xrange(0, offset + 1):
                                a_index = -(1 + offset) + i

                                # if one element doesn't match then bail
                                if B_points[i] != A_points[a_index]:
                                    match = False
                                    break

                            if match:
                                matches += 1
                                changed = True
                                # change tid of B to match A and remove dupes
                                if d:
                                    print "\n> MATCH between TIDS:", A, B
                                    print "First PID:", A_points[-1]
                                    print "\n> Points in match:", offset + 1
                                    raw_input()
                                if not d:
                                    sys.stdout.write("\r" + str(matches))
                                    sys.stdout.flush()

                                del B_points[0:offset + 1]

                                for point in tid_pid:
                                    if point[TID] == B and point[PID] in B_points:
                                        point[TID] = A
                                    elif point[TID] == B and point[PID] not in B_points:
                                        point[TID] = 1000

    # HAVE THEIR BEEN ANY CHANGES TO THE DATA SET AT ALL?
    print "\n> dataset changed:", changed
    if changed:
        print "> Re-running..."
        stitch()


# given a pair of PIDs, work out the angle they make with the vertical
def getAngle(pid1, pid2):

    p1 = None
    p2 = None

    for row in tid_pid:
        if row[PID] == pid1:
            p1 = row
            break

    for row in tid_pid:
        if row[PID] == pid2:
            p2 = row
            break

    dx = float(p2[XC]) - float(p1[XC])
    dy = float(p2[YC]) - float(p1[YC])

    try:
        opp_adj = float(dy / dx)
        theta = math.atan(opp_adj)
        theta = math.degrees(theta)

    except ZeroDivisionError:
        theta = 90.0

    return theta

stitch()

# write the new data back to file
outfile = open('data/data_trajectories.txt', 'w')
for counter in range(0, max_tid + 1):
    for row in tid_pid:
        if int(row[0]) == counter:
            outfile.write(row[TID] + " " + row[XC] + " " + row[YC] + " " +
                          row[FRAME] + " " + row[PID] + "\n")

outfile.close()

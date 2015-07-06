#!/usr/local/bin/python

import sys
import numpy as np

with open("data_trajectories.txt") as datafile:
    data = datafile.read()
    datafile.close()

data = data.split("\n")
data.pop(-1)

# for each pair of trajectories A and B in data
# each point will be (tid,pid)
tid_pid = []
TID = 0
PID = 3

for row in data:
    tid_pid.append(list((row.split(' ')[0], row.split(' ')[1],
                   row.split(' ')[2], row.split(' ')[3])))

tid_list = []

for point in tid_pid:
    if point[TID] not in tid_list:
        tid_list.append(point[TID])

int_list = map(int, tid_list)
max_tid = max(int_list)


def stitch(tid_list, tid_pid):
    changed = False

    # for each trajectory
    for A in tid_list:
        # collect the points into a list
        A_points = []
        for point in tid_pid:
            if point[TID] == A:
                A_points.append(point[PID])

        # for each other trajectory
        for B in tid_list:
            # collect the trajectory into a list
            if A != B:
                B_points = []
                for point in tid_pid:
                    if point[TID] == B:
                        B_points.append(point[PID])

                if len(B_points) > 1 and len(A_points) > 1:

                    # if the last two points of A are the first two points of B
                    if A_points[-1] == B_points[1]:
                        if A_points[-2] == B_points[0]:
                            changed = True
                            print "\n> MATCH between TIDS:", A, B
                            print ">", A_points[-2], B_points[0]
                            print ">", A_points[-1], B_points[1]

                            # change tid of B to match A and and remove dupes
                            print "> deleting points", B_points[0:2], "from", B
                            del B_points[0:2]

                            # do this to original data
                            print "> updating data with stitch"
                            for point in tid_pid:
                                if point[TID] == B and point[PID] in B_points:
                                    point[TID] = A
                                elif point[TID] == B and
                                point[PID] not in B_points:
                                    point[0] = 1000

    # HAVE THEIR BEEN ANY CHANGES TO THE DATA SET AT ALL?
    print "\n> dataset changed:", changed
    if changed:
        print "> Re-running..."
        stitch(tid_list, tid_pid)

stitch(tid_list, tid_pid)

# write the new data back to file
outfile = open('data_trajectories.txt', 'w')
for counter in range(0, max_tid + 1):
    for row in tid_pid:
        if int(row[0]) == counter:
            outfile.write(row[0] + " " + row[1] + " " + row[2] + " " +
                          row[3] + "\n")

outfile.close()

#!/usr/local/bin/python

import sys
import numpy as np

with open(sys.argv[1]) as datafile:
	data = datafile.read()
	datafile.close()

data = data.split("\n")
data.pop(-1)

# for each pair of trajectories A and B in data
# each point will be (tid,pid)
tid_pid = []
TID = 0
PID = 1

for row in data:
	tid_pid.append((row.split(' ')[0], row.split(' ')[3]))

tid_list = []

for point in tid_pid:
	if point[TID] not in tid_list:
		tid_list.append(point[TID])

# for each trajectory
for A in tid_list:
	print "A:", A
	# collect the points into a list
	A_points = []
	for point in tid_pid:
		if point[TID] == A:
			A_points.append(point[PID])

	# for each other trajectory
	for B in tid_list:

		# collect the trajectory into a list
		if A!=B:
			print "-------\nB:", B
			raw_input()
			B_points = []
			for point in tid_pid:
				if point[TID] == B:
					B_points.append(point[PID])

			print A_points[-2],B_points[0]
			print A_points[-1], B_points[1]

			if A_points[-1] == B_points[1]:
				if A_points[-2] == B_points[0]:
					print "MATCH between TIDS:", A, B
	

	# if at least the last two points of A are the first two points of B
		# change tid of B to match A and and remove the duplicate points 
		# can do this to the original data because compound stitches are allowed

# write the new data back to file
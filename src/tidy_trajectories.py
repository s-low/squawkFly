#!/usr/local/bin/python

import sys
import numpy as np

TID = 0
PID = 1
count = 0

with open("kalman_points.txt") as datafile:
	kalman = datafile.read()
	datafile.close()

data = kalman.split('\n')
data.pop(-1)

# each point will be (tid,pid)
tid_pid = []

for row in data:
	tid_pid.append((row.split(' ')[0],row.split(' ')[3]))

# create a list of trajectory id's
tid_list = []

for point in tid_pid:
	if point[TID] not in tid_list:
		tid_list.append(point[TID])

num = len(tid_list)

# for each trajectory (tid A)
for A in tid_list:

	A_points = []
	
	for point in tid_pid:
		if point[TID] == A:
			A_points.append(point[PID])
	# print "\nTRAJECTORY", A, A_points

	# for each other trajectory (tid B)
 	for B in tid_list:
 		trivial = True
 		B_points = []
 		if A != B:

			# for each point in B where B!=A
			for point in tid_pid:
				if point[TID] == B:
					B_points.append(point[PID])
					if point[PID] not in A_points:
						trivial = False

			# if every point from B was in A
			if trivial:
				count += 1
				tid_list.remove(B)
				# print "\nTRIVIAL"
				# print B, B_points
				# print "included in"
				# print A, A_points
				

# write the remaining trajectories to a new file
outfile = open('reduced_trajectories.txt', 'w')

for row in data:
	if row[0] in tid_list:
		outfile.write(row.split(' ')[0] + " " + row.split(' ')[1] + " " + row.split(' ')[2] + " " + row.split(' ')[3]+ "\n")

outfile.close()

print count,"/",num,"removed as trivial"


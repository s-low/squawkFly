#!/usr/local/bin/python

# INPUT: Output.txt containing set of ball candidates for each frame
# format is space delimited three column file eg:
# X    Y    F
# 1    5    1
# 16   3    1
# 1    35   3
# 20   4    4

# OUTPUT: Set of candidate trajectories

import sys
import cv2
import numpy as np 

# KALMAN PARAMETERS
max_dist = 20

with open("output.txt") as datafile:
	data = datafile.read()
	datafile.close()

data = data.split('\n')

all_x = [row.split(' ')[0] for row in data]
all_y = [row.split(' ')[1] for row in data]
all_frames = [row.split(' ')[2] for row in data]

# now translate into frame array
max_frame = int(all_frames[-1])
frame_array = [{} for x in xrange(max_frame+1)]

# for each data point - dump it into the frame of dictionaries
for i in range(0, max_frame+1):
	frame_array[i]["x"] = []
	frame_array[i]["y"] = []

# for each recorded frame
for row in data:	
	x = row.split(' ')[0]
	y = row.split(' ')[1]
	f = int(row.split(' ')[2])

	frame_array[f]["x"].append(x)
	frame_array[f]["y"].append(y)

# frame_array is an array of dictionaries - each containing the dataset for that
# particular frame. 

# FOR each frame F0:
for index, f0 in enumerate(frame_array):
	print "Frame: "+ `index`
	print f0

	# always need two frames of headroom
	if index == max_frame - 1:
		break

	# next frames
	f1 = frame_array[index + 1]
	f2 = frame_array[index + 2]

	# FOR each candidate b in F0:
	for b in f0["x"]:
		b_x = float(f0["x"].pop(0))
		b_y = float(f0["y"].pop(0))

		# FOR each candidate b1 in F1:
		for b1 in f1["x"]:
			b1_x = float(f1["x"].pop(0))
			b1_y = float(f1["y"].pop(0))
		
			# IF separation between b and b+ is small:
			xdiff = abs(b_x - b1_x)
			ydiff = abs(b_y - b1_y)
			sep = ((xdiff**2) + (ydiff**2))**(0.5)
			
			# if sep < max_dist:
				# init kalman filter
				# predict location in F++

				# IF prediction is verified:
					
					# add b, b+, b++ to T_cand
					# update prediction function

				# ELSE:
					
					# IF too many unverified predictions:
						# start a new trajectory

					# But always:
					# Estimate new ball location
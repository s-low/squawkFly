#!/usr/local/bin/python

# IN: Output.txt containing set of ball candidates for each frame
# space delimited three column file eg:

# X    Y    F
# 1    5    1
# 16   3    1
# 1    35   3
# 20   4    4

# OUT: Set of candidate trajectories

import sys
import cv2
import numpy as np 

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

# FOR each frame F:
for index, frame in enumerate(frame_array):
	print index, frame
	
	# FOR each candidate b in F:
	for x_i, x in enumerate(frame["x"]):
		b_x = x 
		b_y = frame["y"][x_i]


		# FOR each candidate b+ in F+:
		for b_plus_i, b_plus in 

			# IF separation between b and b+ is small:
				
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
#!/usr/local/bin/python

# 1D Kalman filter.

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
import cv2.cv as cv
import numpy as np 

from kfilter import KFilter

# KALMAN PARAMETERS
init_dist = 80
verify_distance = 40
max_frame = 0
new_trajectory = True

def verified(corrected_point, next_frame_index):
	next_frame = frame_array[next_frame_index]

	for point_index, point in enumerate(next_frame["x"]):
		cx = float(next_frame["x"][point_index])
		cy = float(next_frame["y"][point_index])
		c = (cx, cy)
		
		if point_is_near_point(corrected_point, c, verify_distance):
			return c

	return False

def point_is_near_point(point1, point2, dist):
	x1 = point1[0]
	y1 = point1[1]	
	x2 = point2[0]
	y2 = point2[1]

	xdiff = float(x1) - float(x2)
	ydiff = float(y1) - float(y2)
	sep = ((xdiff**2) + (ydiff**2)) ** 0.5
	if sep < dist:
		return True
	else: 
		return False

# given a valid pair of nearby points, try to build the next step in their trajectory
def build_trajectory(this_trajectory, kf, frame_index, p0, p1):

	global new_trajectory
	# print "\nBuilding trajectory"
	# print "neck:", p0
	# print "head:", p1
	x = p1[0]
	y = p1[1]

	# predict and correct

	# print "pre\n",np.asarray(kf.getPreState())
	kf.update(x, y)
	corrected = kf.getCorrected()
	predicted = kf.getPredicted()
	# print "post\n",np.asarray(kf.getPostState())

	p2 = None
	p_verification = verified(corrected, frame_index+1)

	if p_verification is not False:
		# add b, b+, b++ to T_cand and re-predict
		print ""
		print "f:"+`frame_index-1`, p0
		print "f:"+`frame_index`, p1
		print "Predicted:",predicted
		print "Corrected:",corrected
		print "VERIFIED by f:"+`frame_index+1`,p_verification

		if new_trajectory:
			this_trajectory.append(p0)
			new_trajectory = False

		this_trajectory.append(p1)
		build_trajectory(this_trajectory, kf, frame_index+1, p1, p_verification)

	elif p_verification == False:
		if new_trajectory == False:
			print "--------------END-----------------"
		# if too_many_misses:
		# 	new trajectory time
		# if not_too_many_misses:
		new_trajectory = True
		pass

	return this_trajectory

# retrieve output of detection system and parse
def get_data(filename):
	global max_frame

	with open(filename) as datafile:
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

	return frame_array


frame_array = get_data(sys.argv[1])
outfile = open('kalman_points.txt', 'w')
trajectories = []

# FOR each frame F0:
for frame_index, f0 in enumerate(frame_array):
	
	# always need two frames of headroom
	if frame_index == max_frame - 1:
		break

	f1 = frame_array[frame_index + 1]
	f2 = frame_array[frame_index + 2]

	# FOR each candidate b in F0:
	for b0_index, b0 in enumerate(f0["x"]):
		
		b0_frame = frame_index
		b0_x = float(f0["x"][b0_index])
		b0_y = float(f0["y"][b0_index])
		b0 = (b0_x, b0_y)

		# FOR each candidate pair of b and b1:
		for b1_index, b1 in enumerate(f1["x"]):
			
			b1_frame = frame_index + 1
			b1_x = float(f1["x"][b1_index])
			b1_y = float(f1["y"][b1_index])
			b1 = (b1_x, b1_y)
		
			# IF separation between b and b+ is small
			xdiff = abs(b0_x - b1_x)
			ydiff = abs(b0_y - b1_y)
			sep = ((ydiff**2) + (xdiff**2)) ** 0.5
			
			if sep < init_dist:

				# init kalman and try to build a single trajectory
				kf = KFilter()
				vx = xdiff
				vy = ydiff

				kf.setPostState(b1[0], b1[1], vx, vy)
	
				this_t = []
				trajectory = build_trajectory(this_t, kf, frame_index+1, b0, b1)
				
				if len(trajectory) != 0:
					trajectories.append(trajectory)

print ""
for ti, trajectory in enumerate(trajectories):
	print "Found trajectory of length:", len(trajectory)
	for point in trajectory:
		outfile.write(`ti` + " " + `point[0]` +" "+ ` point[1]` + "\n")

print np.asarray(kf.getPostState())

outfile.close()
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
from pykalman import KalmanFilter

def verified(corrected_point, next_frame):
	for point in next_frame["x"]:
		cx = next_frame["x"].pop(0)
		cy = next_frame["y"].pop(0)
		c = (cx, cy)
		
		if point_is_near_point(corrected_point, c, 60):
			print "VERIFIED"
			return True

	return False

def point_is_near_point(point1, point2, dist):
	x1 = point1[0]
	y1 = point1[1]	
	x2 = point2[0]
	y2 = point2[1]

	print "P1: ", x1, y1
	print "P2: ",x2, y2	

	xdiff = float(x1) - float(x2)
	ydiff = float(y1) - float(y2)
	sep = ((xdiff**2) + (ydiff**2)) ** 0.5
	print "Sep: ",sep
	if sep < dist:
		return True
	else: 
		return False

# KALMAN PARAMETERS
max_dist = 40

with open("output.txt") as datafile:
	data = datafile.read()
	datafile.close()

outfile = open('kalman_points.txt', 'w')

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


trajectories = []

# FOR each frame F0:
for index, f0 in enumerate(frame_array):
	# print "Frame: "+ `index`
	# print f0

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
			sep = ((ydiff**2) + (xdiff**2)) ** 0.5
			
			if sep < max_dist:
				this_trajectory = []
				print "Initalising Kalman Filter"
				# INITIALISE KALMAN FILTER
				# state vec (x, y, v_x, v_y)
				# measure vec (z_x, z_y)

				kf = cv.CreateKalman(4, 2, 0)
				state = cv.CreateMat(4, 1, cv.CV_32FC1)
				proc_noise  = cv.CreateMat(4, 1, cv.CV_32FC1)
				measurement = cv.CreateMat(2, 1, cv.CV_32FC1)
				
				# transition matrix init 
				for j in range(4):
					for k in range(4):
						kf.transition_matrix[j,k] = 0
					kf.transition_matrix[j,j] = 1

				kf.transition_matrix[0,2] = 1
				kf.transition_matrix[1,3] = 1

				# measurement matrix init
				cv.SetIdentity(kf.measurement_matrix)

				processNoiseCovariance = 1e-4
				measurementNoiseCovariance = 1e-1
				errorCovariancePost= 0.1

				cv.SetIdentity(kf.process_noise_cov, cv.RealScalar(processNoiseCovariance))
				cv.SetIdentity(kf.measurement_noise_cov, cv.RealScalar(measurementNoiseCovariance))
				cv.SetIdentity(kf.error_cov_post, cv.RealScalar(errorCovariancePost))

				# predict location in f2 from state f1
				measurement[0, 0] = b1_x
				measurement[1, 0] = b1_y

				predicted = cv.KalmanPredict(kf)
				corrected = cv.KalmanCorrect(kf, measurement)

				# print "State: \n", np.asarray(s_vector[:,:])
				# print "\nMeasurement:\n", np.asarray(measurement[:,:])
				# print "\nPrediction: \n", np.asarray(predicted[:,:])
				# print "\nCorrected: \n",  np.asarray(corrected[:,:])

				corrected_point = (corrected[0,0], corrected[1,0])

				# IF prediction is verified
				if verified(corrected_point, f2):
					# add b, b+, b++ to T_cand
					this_trajectory.append(corrected_point)

					# update prediction function

				trajectories.append(this_trajectory)
				# ELSE:
					
					# IF too many unverified predictions:
						# start a new trajectory

					# But always:
					# Estimate new ball location

for trajectory in trajectories:
	print trajectory
	for point in trajectory:
		outfile.write(`point[0]` +" "+ ` point[1]` + "\n")

outfile.close()
#!/usr/local/bin/python

''' kalman.py

2D Kalman filter-based trajectory segmentation

INPUT: The set of ball detections from detect.py

in format: x y frame point_id

OUTPUT: A set of candidate trajectories to data_trajectories.txt

in format: trajectory_id x y frame point_id

*arg1* = optional infile, otherwise data/data_detections.txt
*arg2* = optional outfile, otherwise data/data_trajectories.txt

'''

import sys
import cv2
import cv2.cv as cv
import numpy as np
import plotting as plot
import matplotlib.pyplot as plt

# Trajectory generation / verification parameters
init_dist = 400
denom = 2.5
min_v_dist = 6

# Kalman covariances and system equation parameters
Sensor_Cov = 10
PN_Cov = 4
horizontal_acc_const = -0.04

# Program markers
max_frame = 0
max_length = 0
new_trajectory = True
n_miss = 0
max_misses = 7
min_length = 8

# debug mode
d = False
graphs = False
predictions = []
detections = []
corrections = []


# create a fresh kalman filter object using OpenCV and return it
def KalmanFilter():
    kf = cv.CreateKalman(6, 2, 0)

    '''
    init the prediction/evolution/transition matrix
    | 1  0  1  0 .5  0 | x  |   | x  + vx  + .5ax |
    | 0  1  0  1  0 .5 | y  | = | y  + vy  + .5ay |
    | 0  0  1  0  1  0 | vx |   | vx + ax         |
    | 0  0  0  1  0  1 | vy |   | vy + ay         |
    | 0  0  k  0  0  0 | ax |   |   k*vx          |
    | 0  0  0  0  0  1 | ay |   |    ay           |
    '''

    # diagonals
    for j in range(6):
        for k in range(6):
            kf.transition_matrix[j, k] = 0
        kf.transition_matrix[j, j] = 1

    # x + vx + 0.5ax
    kf.transition_matrix[0, 2] = 1
    kf.transition_matrix[0, 4] = 0.5

    # y + vy + 0.5ay
    kf.transition_matrix[1, 3] = 1
    kf.transition_matrix[1, 5] = 0.5

    # vx + ax
    kf.transition_matrix[2, 4] = 1

    # vy + ay
    kf.transition_matrix[3, 5] = 1

    # predict ax = k * vx
    kf.transition_matrix[4, 4] = 0
    kf.transition_matrix[4, 2] = horizontal_acc_const

    '''
    measurement matrix H: mean = H * state
    | 1 0 | x |   | x |
    | 0 1 | y | = | y |
    '''
    kf.measurement_matrix[0, 0] = 1
    kf.measurement_matrix[1, 1] = 1

    # process noise cov matrix Q: models the EXTERNAL uncertainty
    cv.SetIdentity(kf.process_noise_cov, PN_Cov)

    # measurement noise cov matrix R: covariance of SENSOR noise
    cv.SetIdentity(kf.measurement_noise_cov, Sensor_Cov)

    '''
    error estimate covariance matrix P: relates the correlation of state vars
    priori: before measurement
    posteriori: after measurement
    diagonals are all 1. x-vy and y-vy also correlated.

    | xx  xy  xvx  xvy  xax  xay  |   | 1 0 1 0 0 0 |
    | yx  yy  yvx  yvy  yax  yay  |   | 0 1 0 1 0 0 |
    | vxx vxy vxvx vxvy vxax vxay | = | 1 0 1 0 0 0 |
    | vyx vyy vyvx vyvy vyax vyay |   | 0 1 0 1 0 0 |
    | axx axy axvx axvy axax axay |   | 0 0 0 0 1 0 |
    | ayx ayy ayvx ayvy ayax ayay |   | 0 0 0 0 0 1 |
    '''
    cv.SetIdentity(kf.error_cov_post, 1)
    kf.error_cov_post[0, 2] = 1
    kf.error_cov_post[1, 3] = 1
    kf.error_cov_post[2, 0] = 1
    kf.error_cov_post[3, 1] = 1

    return kf


# Manually set the KF post-state to initalise it
def setPostState(x, y, vx, vy, ax, ay):
    global kf
    kf.state_post[0, 0] = x
    kf.state_post[1, 0] = y
    kf.state_post[2, 0] = vx
    kf.state_post[3, 0] = vy
    kf.state_post[4, 0] = ax
    kf.state_post[5, 0] = ay


# kalman filter predict and return it as a tuple
def predict(kf):
    predicted = cv.KalmanPredict(kf)
    return (predicted[0, 0], predicted[1, 0],
            predicted[2, 0], predicted[3, 0],
            predicted[4, 0], predicted[5, 0])


# KF correct and return it as a tuple
def correct(kf, x, y):
    global measurement
    measurement[0, 0] = x
    measurement[1, 0] = y

    corrected = cv.KalmanCorrect(kf, measurement)
    return (corrected[0, 0], corrected[1, 0],
            corrected[2, 0], corrected[3, 0],
            corrected[4, 0], corrected[5, 0])


# Check if the prediction is verified by a detection in the next frame
def verified(predicted_point, next_frame_index, v_distance):
    try:
        next_frame = frame_array[next_frame_index]
    except IndexError:
        return False

    # init the measured sep to an arbitrarily high value
    min_sep = v_distance
    vpoint = None

    # for each candidate in the next frame, check the distance between the pair
    for point_index, point in enumerate(next_frame["x"]):
        cx = float(next_frame["x"][point_index])
        cy = float(next_frame["y"][point_index])
        c_pid = int(next_frame["pid"][point_index])
        c_frame = next_frame_index

        # POINT: X / Y / FRAME / PID
        c = (cx, cy, c_frame, c_pid)

        # Check the separation against the current verify distance
        vflag, sep = point_is_near_point(predicted_point, c, v_distance)

        # a point may be verified by several points, but we want the best one.
        if vflag is True and sep < min_sep:
            min_sep = sep
            vpoint = c

    if vpoint is not None:
        return vpoint
    else:
        return False


# Is distance between point A and point B less than C
def point_is_near_point(point1, point2, dist):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    xdiff = float(x1) - float(x2)
    ydiff = float(y1) - float(y2)
    sep = ((xdiff ** 2) + (ydiff ** 2)) ** 0.5
    if sep < dist:
        return True, sep
    else:
        return False, sep


''' build_trajectory()

main recursive trajectory generation method.

given a valid pair of nearby points in sequential frames, initalise the
kalman filter and try to follow that trajectory forwards in time.

Current two points are referred to as the HEAD and ARM

o-----o-----o-----o-----o-----x
                  HEAD  ARM   PREDICTED

'''
def build_trajectory(this_trajectory, bridge, kf, frame_index, p0, p1, real):

    global predictions
    global detections
    global corrections
    global d
    global new_trajectory
    global n_miss
    postState = kf.state_post
    preState = kf.state_pre

    if d and new_trajectory:
        print "\nNEW"

    if d:
        print "\ntrajectory:\n", this_trajectory
        print "Head:", p0
        print "Arm:", p1
        print "Post state:\n", np.asarray(postState[:, :])
        print "Pre state:\n", np.asarray(preState[:, :])

    # PREDICT location of next point
    predicted = predict(kf)
    predictions.append((predicted[0], predicted[1]))
    if d:
        print "Predicted:", predicted

    # Set verifying distance to fraction of current speed, or a minumum value
    v_dist = (((postState[2, 0] ** 2) + (postState[3, 0] ** 2)) ** 0.5) / denom
    if v_dist < min_v_dist:
        v_dist = min_v_dist

    # MEASURE location of verifying point
    p_verification = verified(predicted, frame_index + 1, v_dist)

    if d:
        print "Verified by:", p_verification

    if p_verification is False:
        n_miss += 1

        # If we've made too many unverified predictions, give up.
        if n_miss >= max_misses:
            if d:
                print "Bridge too far. End at last verified point."
            n_miss = 0
            new_trajectory = True
            bridge = []
            predictions = []

        # Otherwise, have another crack
        else:
            # keep predicting from the unverified corrected point
            unverified = (predicted[0], predicted[1], frame_index + 1, 1000)
            if d:
                print "Append predicted point to bridge:", unverified
            bridge.append(unverified)

            if graphs:
                plt.plot(all_x, all_y, '.')

                x = [t[0] for t in this_trajectory]
                y = [t[1] for t in this_trajectory]
                plt.plot(x, y, 'r.')
                plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'm.')

                x = [p[0] for p in detections]
                y = [p[1] for p in detections]
                plt.plot(x, y, 'r')

                x = [p[0] for p in corrections]
                y = [p[1] for p in corrections]
                plt.plot(x, y, 'b')

                x = [p[0] for p in predictions]
                y = [p[1] for p in predictions]

                plt.plot(x, y, 'g.')
                plt.show()

            # Recursive call from new trajectory
            this_trajectory = build_trajectory(
                this_trajectory, bridge, kf, frame_index + 1, p1, unverified, False)

    # PREDICTION WAS VERIFED BY MEASUREMENT
    else:
        # CORRECT filter against the verifying (but noisy) measurement
        x = p_verification[0]
        y = p_verification[1]
        corrected = correct(kf, x, y)
        corrections.append((corrected[0], corrected[1]))

        if d:
            print "Corrected against P_ver:", corrected

        # if a brand new trajectory, add the initialising points too
        if new_trajectory:
            this_trajectory.append(p0)
            # only append p1 if it was a real point
            if real:
                this_trajectory.append(p1)
            new_trajectory = False

        # if a bridge of unverifieds was needed to get here, reset it to zero
        if len(bridge) != 0:
            bridge = []
            n_miss = 0

        # add verifying point to trajectory + continue
        this_trajectory.append(p_verification)
        detections.append((x, y))

        if graphs:
            plt.plot(all_x, all_y, '.')
            x = [t[0] for t in this_trajectory]
            y = [t[1] for t in this_trajectory]
            plt.plot(x, y, 'r.')
            plt.plot([p0[0], p1[0]], [p0[1], p1[1]], 'm.')

            x = [p[0] for p in detections]
            y = [p[1] for p in detections]
            plt.plot(x, y, 'r')

            x = [p[0] for p in corrections]
            y = [p[1] for p in corrections]
            plt.plot(x, y, 'b')

            # x = [p[0] for p in predictions]
            # y = [p[1] for p in predictions]
            p = predictions[-1]
            x = p[0]
            y = p[1]
            plt.plot(x, y, 'g.')
            plt.show()

        # RECURSIVE CALL WITH NEW HEAD AND ARM
        this_trajectory = build_trajectory(this_trajectory, bridge, kf,
                                           frame_index + 1, p1,
                                           p_verification,
                                           True)
        detections = []
        predictions = []
        corrections = []
    return this_trajectory


# retrieve output of detection system and parse it
def get_data(filename):
    global max_frame

    with open(filename) as datafile:
        data = datafile.read()
        datafile.close()

    data = data.split('\n')

    # Gobble blank at EOF if exists
    if data[-1] in ['\n', '\r\n', '']:
        data.pop(-1)

    # data_detections in form: X / Y / FRAME / PID
    all_x = [row.split()[0] for row in data]
    all_y = [row.split()[1] for row in data]
    all_frames = [row.split()[2] for row in data]
    all_pid = [row.split()[3] for row in data]

    # now translate into 'frame_array' structure
    # each element of array represents a frame, and holds all the detections
    # in that particular frame
    max_frame = int(all_frames[-1])
    frame_array = [{} for x in xrange(max_frame + 1)]

    # create an array of dictionaries, one element for each frame
    for i in range(0, max_frame + 1):
        frame_array[i]["x"] = []
        frame_array[i]["y"] = []
        frame_array[i]["pid"] = []

    # for each detection, get x/y/pid and dump it into the appropriate dict
    for row in data:
        x = row.split()[0]
        y = row.split()[1]
        f = int(row.split()[2])
        p_id = row.split()[3]

        frame_array[f]["x"].append(x)
        frame_array[f]["y"].append(y)
        frame_array[f]["pid"].append(p_id)

    return frame_array, all_x, all_y


'''
-------------------------------------------------------------------------------
-------------------------Main program begins here------------------------------
-------------------------------------------------------------------------------
'''

print "----------KALMAN.PY------------"

try:
    infilename = sys.argv[1]
    print "Getting detections:", infilename
except IndexError:
    infilename = 'data/data_detections.txt'

try:
    outfilename = sys.argv[2]
except IndexError:
    outfilename = 'data/data_trajectories.txt'

frame_array, all_x, all_y = get_data(infilename)
outfile = open(outfilename, 'w')
trajectories = []

# Initialise the state/measurement/prediction/correction for KF
state = cv.CreateMat(6, 1, cv.CV_32FC1)
measurement = cv.CreateMat(2, 1, cv.CV_32FC1)
predicted = None
corrected = None

# FOR each frame F0:
for frame_index, f0 in enumerate(frame_array):

    # always need two frames of headroom to avoid indexError
    if frame_index == max_frame - 1:
        break

    f1 = frame_array[frame_index + 1]
    f2 = frame_array[frame_index + 2]

    # FOR each point b in F0:
    for b0_index, b0 in enumerate(f0["x"]):

        b0_frame = frame_index
        b0_x = float(f0["x"][b0_index])
        b0_y = float(f0["y"][b0_index])
        b0_pid = int(f0["pid"][b0_index])

        # POINT: X / Y / FRAME / PID
        b0 = (b0_x, b0_y, b0_frame, b0_pid)

        # FOR each point pair of b and b1:
        for b1_index, b1 in enumerate(f1["x"]):

            b1_frame = frame_index + 1
            b1_x = float(f1["x"][b1_index])
            b1_y = float(f1["y"][b1_index])
            b1_pid = int(f1["pid"][b1_index])

            # POINT: X / Y / FRAME / PID
            b1 = (b1_x, b1_y, b1_frame, b1_pid)

            # IF separation between b and b+ is small
            xdiff = b1_x - b0_x
            ydiff = b1_y - b0_y
            sep = ((ydiff ** 2) + (xdiff ** 2)) ** 0.5

            # If two points are closer than the initisation distance
            if sep < init_dist:

                # init new kalman filter and try to build a single trajectory
                kf = KalmanFilter()
                vx = xdiff
                vy = ydiff

                if d:
                    print "\n-------- INIT Filter --------"
                    print "Points:", b0, b1

                # Manually initialise state with guess at speed as well
                setPostState(b1[0], b1[1], vx, vy, 0, 0)
                if d:
                    print "Post state set:", b1[0], b1[1], vx, vy, 0, 0

                this_t = []
                bridge = []
                trajectory = build_trajectory(
                    this_t, bridge, kf, frame_index + 1, b0, b1, True)

                if len(trajectory) != 0:
                    trajectories.append(trajectory)
                    if len(trajectory) > max_length:
                        max_length = len(trajectory)

print ""
count = 0
ti = 0

# write: TID / X / Y / FRAME / PID
for ti, trajectory in enumerate(trajectories):
    if len(trajectory) > min_length:
        count += 1
        for p in trajectory:
            outfile.write(str(count) + " " + str(p[0]) + " " +
                          str(p[1]) + " " + str(p[2]) + " " + str(p[3]) + "\n")

print "> Found", ti, "trajectories"
print ">", count, "are longer than", min_length, "points"
print "> Most Detections:", max_length
print "> written to: ", outfilename

outfile.close()

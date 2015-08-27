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
import plotting as plot
import matplotlib.pyplot as plt

# Kalman Parameters
init_dist = 400
verify_distance = 50

# Program markers
max_frame = 0
max_length = 0
new_trajectory = True
n_miss = 0
max_misses = 6
min_length = 2

# debug mode
d = False
predictions = []
detections = []
corrections = []


# create a fresh kalman filter object and return it
def KalmanFilter():
    kf = cv.CreateKalman(6, 2, 0)

    '''
    init the prediction/evolution/transition matrix
    | 1  0  1  0 .5  0 | x  |   | x  + vx  + .5ax |
    | 0  1  0  1  0 .5 | y  | = | y  + vy  + .5ay |
    | 0  0  1  0  1  0 | vx |   | vx + ax         |
    | 0  0  0  1  0  1 | vy |   | vy + ay         |
    | 0  0  0  0  1  0 | ax |   |    ax           |
    | 0  0  0  0  0  1 | ay |   |    ay           |
    '''

    # diagonals
    for j in range(6):
        for k in range(6):
            kf.transition_matrix[j, k] = 0
        kf.transition_matrix[j, j] = 1

    # off-diagonals
    kf.transition_matrix[0, 2] = 1
    kf.transition_matrix[0, 4] = 0.5
    kf.transition_matrix[1, 3] = 1
    kf.transition_matrix[1, 5] = 0.5
    kf.transition_matrix[2, 4] = 1
    kf.transition_matrix[3, 5] = 1

    print np.asarray(kf.transition_matrix)

    '''
    measurement matrix H: mean = H * state
    | 1 0 | x |   | x |
    | 0 1 | y | = | y |
    '''
    kf.measurement_matrix[0, 0] = 1
    kf.measurement_matrix[1, 1] = 1

    # process noise cov matrix Q: models the EXTERNAL uncertainty
    cv.SetIdentity(kf.process_noise_cov, cv.RealScalar(4))

    # measurement noise cov matrix R: covariance of SENSOR noise
    cv.SetIdentity(kf.measurement_noise_cov, cv.RealScalar(10))

    '''
    error estimate covariance matrix P: relates the correlation of state vars
    priori: before measurement
    posteriori: after measurement
    | xx  xy  xvx  xvy  xax  xay  |   | 1 0 1 0 0 0 |
    | yx  yy  yvx  yvy  yax  yay  |   | 0 1 0 1 0 0 |
    | vxx vxy vxvx vxvy vxax vxay | = | 1 0 1 0 0 0 |
    | vyx vyy vyvx vyvy vyax vyay |   | 0 1 0 1 0 0 |
    | axx axy axvx axvy axax axay |   | 0 0 0 0 1 0 |
    | ayx ayy ayvx ayvy ayax ayay |   | 0 0 0 0 0 1 |
    '''
    cv.SetIdentity(kf.error_cov_post, cv.RealScalar(1))
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


# kalman filter predict and return as tuple
def predict(kf):
    predicted = cv.KalmanPredict(kf)
    return (predicted[0, 0], predicted[1, 0],
            predicted[2, 0], predicted[3, 0],
            predicted[4, 0], predicted[5, 0])


# KF correct and return as tuple
def correct(kf, x, y):
    global measurement
    measurement[0, 0] = x
    measurement[1, 0] = y

    corrected = cv.KalmanCorrect(kf, measurement)
    return (corrected[0, 0], corrected[1, 0],
            corrected[2, 0], corrected[3, 0],
            corrected[4, 0], corrected[5, 0])


def verified(corrected_point, next_frame_index, v_distance):
    try:
        next_frame = frame_array[next_frame_index]
    except IndexError:
        return False

    # init the measured sep to an arbitrarily high value
    min_sep = v_distance
    vpoint = None

    # for each point in the next frame, check the distance between the pair
    for point_index, point in enumerate(next_frame["x"]):
        cx = float(next_frame["x"][point_index])
        cy = float(next_frame["y"][point_index])
        c_pid = int(next_frame["pid"][point_index])
        c_frame = next_frame_index

        # POINT: X / Y / FRAME / PID
        c = (cx, cy, c_frame, c_pid)

        vflag, sep = point_is_near_point(corrected_point, c, v_distance)

        # a point may be verified by several points. we want the best one.
        if vflag is True and sep < min_sep:
            min_sep = sep
            vpoint = c

    if vpoint is not None:
        return vpoint
    else:
        return False


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


# given a valid pair of nearby points, try to build the next step in trajectory
def build_trajectory(this_trajectory, bridge, kf, frame_index, p0, p1, real):

    global predictions
    global detections
    global corrections
    global d
    global new_trajectory
    global n_miss
    postState = kf.state_post
    preState = kf.state_pre

    # if p1[3] == 71:
    #     d = True

    if d and new_trajectory:
        print "\nNEW"

    if d:
        print "\ntrajectory:\n", this_trajectory
        print "Head:", p0
        print "Arm:", p1
        print "Post state:\n", np.asarray(postState[:, :])
        print "Pre state:\n", np.asarray(preState[:, :])

    # PREDICT location of branch
    predicted = predict(kf)
    predictions.append((predicted[0], predicted[1]))
    if d:
        print "Predicted:", predicted

    # MEASURE location of verifying point
    # v_dist is half current speed
    v_dist = (((postState[2, 0] ** 2) + (postState[3, 0] ** 2)) ** 0.5) / 2.5
    p_verification = verified(predicted, frame_index + 1, v_dist)

    if d:
        print "Verified by:", p_verification

    if p_verification is False:
        n_miss += 1

        if n_miss >= max_misses:
            if d:
                print "Bridge too far. End at last verified point."
            n_miss = 0
            new_trajectory = True
            bridge = []

        else:
            # keep predicting from the unverified corrected point
            # POINT: X / Y / FRAME / PID
            unverified = (predicted[0], predicted[1], frame_index + 1, 1000)
            new_trajectory = False
            if d:
                print "Append predicted point to bridge:", unverified
            bridge.append(unverified)

            # x = [p[0] for p in detections]
            # y = [p[1] for p in detections]
            # plt.plot(x, y, 'r')

            # x = [p[0] for p in corrections]
            # y = [p[1] for p in corrections]
            # plt.plot(x, y, 'b')

            # x = [p[0] for p in predictions]
            # y = [p[1] for p in predictions]
            # plt.plot(x, y, 'g')
            # plt.show()

            this_trajectory = build_trajectory(
                this_trajectory, bridge, kf, frame_index + 1, p1, unverified, False)

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

        # if a bridge of unverifieds was needed reset it
        if len(bridge) != 0:
            bridge = []
            n_miss = 0

        # add verified point to trajectory + continue
        this_trajectory.append(p_verification)
        detections.append((x, y))

        # x = [p[0] for p in detections]
        # y = [p[1] for p in detections]
        # plt.plot(x, y, 'r')

        # x = [p[0] for p in corrections]
        # y = [p[1] for p in corrections]
        # plt.plot(x, y, 'b')

        # x = [p[0] for p in predictions]
        # y = [p[1] for p in predictions]
        # plt.plot(x, y, 'g')
        # plt.show()

        this_trajectory = build_trajectory(this_trajectory, bridge, kf,
                                           frame_index + 1, p1,
                                           p_verification,
                                           True)
        detections = []
        predictions = []
        corrections = []
    return this_trajectory


# see if a full trajectory can be extended backwards to it's true source
def checkForRoot(trajectory):
    p0 = trajectory[0]
    p1 = trajectory[1]
    vx = p0[0] - p1[0]
    vy = p0[1] - p1[1]

    tx = p0[0] + vx
    ty = p0[1] + vy


# retrieve output of detection system and parse
def get_data(filename):
    global max_frame

    with open(filename) as datafile:
        data = datafile.read()
        datafile.close()

    data = data.split('\n')

    # get rid of any empty line at the end of file
    if data[-1] in ['\n', '\r\n', '']:
        data.pop(-1)

    # data_detections in form: X / Y / FRAME / PID
    all_x = [row.split()[0] for row in data]
    all_y = [row.split()[1] for row in data]
    all_frames = [row.split()[2] for row in data]
    all_pid = [row.split()[3] for row in data]

    # now translate into 'frame array' (each element is a frame)
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

    return frame_array


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

frame_array = get_data(infilename)
outfile = open(outfilename, 'w')
trajectories = []

# create OpenCV Kalman object
state = cv.CreateMat(6, 1, cv.CV_32FC1)
measurement = cv.CreateMat(2, 1, cv.CV_32FC1)
predicted = None
corrected = None

# FOR each frame F0:
for frame_index, f0 in enumerate(frame_array):

    # always need two frames of headroom
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

            if sep < init_dist:

                # init new kalman filter and try to build a single trajectory
                kf = KalmanFilter()
                vx = xdiff
                vy = ydiff

                if d:
                    print "\n-------- INIT Filter --------"
                    print "Points:", b0, b1

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

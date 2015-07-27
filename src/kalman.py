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

# Kalman Parameters
init_dist = 70
verify_distance = 50

# Program markers
max_frame = 0
max_length = 0
new_trajectory = True
n_miss = 0
max_misses = 5


def verified(corrected_point, next_frame_index):
    try:
        next_frame = frame_array[next_frame_index]
    except IndexError:
        return False

    # init current verifying sep to an arbitrarily high value
    min_sep = verify_distance
    vpoint = None

    # for each point in the next frame, check the distance between the pair
    for point_index, point in enumerate(next_frame["x"]):
        cx = float(next_frame["x"][point_index])
        cy = float(next_frame["y"][point_index])
        c_pid = int(next_frame["pid"][point_index])
        c = (cx, cy, c_pid)

        vflag, sep = point_is_near_point(corrected_point, c, verify_distance)

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
def build_trajectory(this_trajectory, bridge, kf, frame_index, p0, p1):

    global new_trajectory
    global n_miss

    x = p1[0]
    y = p1[1]

    # predict and correct
    kf.update(x, y)
    corrected = kf.getCorrected()
    predicted = kf.getPredicted()

    # verify
    p_verification = verified(corrected, frame_index + 1)

    if p_verification is not False:
        # print ""
        # print "f:"+`frame_index-1`, p0
        # print "f:"+`frame_index`, p1
        # print "Predicted:",predicted
        # print "Corrected:",corrected
        # print "VERIFIED by f:"+`frame_index+1`,p_verification

        # if brand new, add the very first point too
        if new_trajectory:
            this_trajectory.append(p0)
            new_trajectory = False

        # if a bridge of unverifieds was needed, add it
        if len(bridge) != 0:
            this_trajectory = this_trajectory + bridge
            bridge = []

        # add verified point to trajectory + continue
        this_trajectory.append(p1)

        build_trajectory(
            this_trajectory, bridge, kf, frame_index + 1, p1, p_verification)

    elif p_verification is False:
        n_miss += 1

        if n_miss >= max_misses:
            # print "--------------END-----------------"
            n_miss = 0
            new_trajectory = True
            bridge = []

        elif n_miss < max_misses:
            # keep predicting from the unverified corrected point
            bridge.append(p1)
            unverified = (corrected[0], corrected[1])
            build_trajectory(
                this_trajectory, bridge, kf, frame_index + 1, p1, unverified)

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
    all_index = [row.split(' ')[3] for row in data]

    # now translate into frame array
    max_frame = int(all_frames[-1])
    frame_array = [{} for x in xrange(max_frame + 1)]

    # for each data point - dump it into the frame of dictionaries
    for i in range(0, max_frame + 1):
        frame_array[i]["x"] = []
        frame_array[i]["y"] = []
        frame_array[i]["pid"] = []

    # for each recorded frame
    for row in data:
        x = row.split(' ')[0]
        y = row.split(' ')[1]
        f = int(row.split(' ')[2])
        p_id = row.split(' ')[3]

        frame_array[f]["x"].append(x)
        frame_array[f]["y"].append(y)
        frame_array[f]["pid"].append(p_id)

    return frame_array


frame_array = get_data('data/data_detections.txt')
outfile = open('data/data_trajectories.txt', 'w')
trajectories = []

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
        b0 = (b0_x, b0_y, b0_pid)

        # FOR each point pair of b and b1:
        for b1_index, b1 in enumerate(f1["x"]):

            b1_frame = frame_index + 1
            b1_x = float(f1["x"][b1_index])
            b1_y = float(f1["y"][b1_index])
            b1_pid = int(f1["pid"][b1_index])
            b1 = (b1_x, b1_y, b1_pid)

            # IF separation between b and b+ is small
            xdiff = abs(b0_x - b1_x)
            ydiff = abs(b0_y - b1_y)
            sep = ((ydiff ** 2) + (xdiff ** 2)) ** 0.5

            if sep < init_dist:

                # init kalman and try to build a single trajectory
                kf = KFilter()
                vx = xdiff
                vy = ydiff

                kf.setPostState(b1[0], b1[1], vx, vy)

                this_t = []
                bridge = []
                trajectory = build_trajectory(
                    this_t, bridge, kf, frame_index + 1, b0, b1)

                if len(trajectory) != 0:
                    trajectories.append(trajectory)
                    if len(trajectory) > max_length:
                        max_length = len(trajectory)

print ""
count = 0
min_length = 2
ti = 0

# write tid / x / y / pid
# but needs to be tid / x / y / pid / FRAME
for ti, trajectory in enumerate(trajectories):
    if len(trajectory) > min_length:
        count += 1
        for point in trajectory:
            outfile.write(str(count) + " " + str(point[0]) + " " +
                          str(point[1]) + " " + str(point[2]) + "\n")

print "> Found", ti, "trajectories"
print ">", count, "are longer than", min_length, "points"
print "> Longest is", max_length, "long"
print "> written to data_trajectories.txt\n"

outfile.close()

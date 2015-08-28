#!/usr/local/bin/python

'''
Calibrate.py

Supplied with either a video file or a sequence of photographs of a calibration
pattern, retrieve the camera intrinsics and write to file if supplied.

arg1 = input video or dir of photographs
arg2 = designated outfile destination *optional*

Note that if an image sequence is supplied (directory of every frame in a
video) then it will be interpreted as a dir of photographs and every frame will
be used up to a maximum of 50.
'''


import sys
import cv2
import cv2.cv as cv
import numpy as np
import glob
import os

# default to showing the calibration images
view = True
try:
    if sys.argv[3] == 'suppress':
        view = False
except IndexError:
    pass

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare (3D) object points (x,y,z): (0,0,0), (1,0,0) ....,(6,5,0)
# using unit squares initially
objp = np.zeros((6 * 9, 3), np.float32)
# just a shortcut to the desired coords
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d points in real space
imgpoints = []  # 2d points in image plane
images = []
num_images = 0

source = sys.argv[1]

# if the source is a directory
if os.path.isdir(source):
    is_video = False
    folder = source + '/*.png'
    print "Calibrate from images:", folder
    images = glob.glob(folder)

# if the source is a file (should be video)
elif os.path.isfile(source):

    is_video = True
    print "Calibrate from video:", source
    cap = cv2.VideoCapture(source)
    count = 0

    while(cap.isOpened()):
        ret, img = cap.read()
        count += 1
        # take every 30th frame (roughly a second at 30FPS)
        if count % 30 == 0:
            num_images += 1
            images.append(img)
            sys.stdout.write("\rGetting images: " + str(num_images))
            sys.stdout.flush()
            if num_images > 49:
                break

    cap.release()

else:
    print "WARN: Neither a file nor a directory."
    sys.exit()

num_images = 0
success_count = 0

for image in images:
    num_images += 1

    if is_video:
        img = image
    else:
        img = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(
        gray, (9, 6),
        flags=cv.CV_CALIB_CB_FILTER_QUADS | cv.CV_CALIB_CB_ADAPTIVE_THRESH)

    # If found, add object points, image points (after refining them)
    if ret is True:
        success_count += 1
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        if view:
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('success', img)
            cv2.waitKey(30)

    if num_images > 50:
        print "> 50 calibration images processed. Stopping."
        break

err, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

nodist = dist
nodist[0][0] = 0
nodist[0][1] = 0
nodist[0][2] = 0
nodist[0][3] = 0
nodist[0][4] = 0

no_dist = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, nodist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    no_dist += error


np.set_printoptions(precision=3, suppress=True)
print success_count, "/", len(images), "successful detections"
print "calibration matrix: \n", mtx
print "distortion:\n", dist
print "avg returned reprojection error:", err
print "avg calculated projection error: ", mean_error / len(objpoints)
print "and assuming no distortion:", no_dist / len(objpoints)

try:
    outfile = open(sys.argv[2], 'w')
    outfile.write(
        str(mtx[0, 0]) + ' ' + str(mtx[0, 2]) + ' -' + str(mtx[1, 2]))
    outfile.close()

except IndexError:
    pass

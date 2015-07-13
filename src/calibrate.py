#!/usr/local/bin/python

# Calibration test script. Retrieve camera matrix based on a checkerboard
# pattern.

import cv2
import cv2.cv as cv
import numpy as np
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare (3D) object points (x,y,z): (0,0,0), (1,0,0) ....,(6,5,0)
# using unit squares initially
objp = np.zeros((6 * 9, 3), np.float32)
# just a shortcut to the desired coords
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d points in real space
imgpoints = []  # 2d points in image plane

images = glob.glob('res/lgg3/*.png')

for image in images:
    print "new image"
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('current', gray)
    # cv2.waitKey()
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret is True:
        print "found corners"
        objpoints.append(objp)

        # corners = cv2.cornerSubPix(
        # gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        cv2.imshow('sucess', img)
        cv2.waitKey()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (1280, 800), None, None)
# gray.shape[::-1]

mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print "camera matrix: \n", mtx
print "average reprojection error: ", mean_error / len(objpoints)

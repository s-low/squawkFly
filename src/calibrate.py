#!/usr/local/bin/python

# Calibration test script. Retrieve camera matrix based on a checkerboard
# pattern.

import cv2
import cv2.cv as cv
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d points in real space
imgpoints = []  # 2d points in image plane

img = cv2.imread("res/left03.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

# If found, add object points, image points (after refining them)
if ret is True:
    objpoints.append(objp)

    # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners)

    # Draw and display the corners
    cv2.drawChessboardCorners(img, (7, 6), corners, ret)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#!/usr/local/bin/python

# credit jan erik solem
# http://www.janeriksolem.net/2014/05/how-to-calibrate-camera-with-opencv-and.html

import numpy as np
import cv2

# copy parameters to arrays
K = np.array(
    [[1091.228, 0, 643.93], [0, 1091.159, 412.837], [0, 0, 1]])
# just use first two terms (no translation)
d = np.array([0.006, 0.558, 0, 0, 0])

# read one of your images
img = cv2.imread("res/g3/11.png")
h, w = img.shape[:2]

# undistort
newcamera, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)
newimg = cv2.undistort(img, K, d, None, newcamera)

cv2.imwrite("res/g3/original.jpg", img)
cv2.imwrite("res/g3/undistorted.jpg", newimg)

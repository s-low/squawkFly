#!/usr/local/bin/python

import cv2
import cv2.cv as cv
import numpy as np

# object points
obj_pts = np.zeros((9, 3), dtype='float32')
obj_pts[0] = np.array([40, 40, 0], dtype='float32')
obj_pts[1] = np.array([50, 40, 0], dtype='float32')
obj_pts[2] = np.array([60, 40, 0], dtype='float32')
obj_pts[3] = np.array([40, 50, 0], dtype='float32')
obj_pts[4] = np.array([50, 50, 0], dtype='float32')
obj_pts[5] = np.array([60, 50, 0], dtype='float32')
obj_pts[6] = np.array([40, 60, 0], dtype='float32')
obj_pts[7] = np.array([50, 60, 0], dtype='float32')
obj_pts[8] = np.array([60, 60, 0], dtype='float32')

# image points
img_pts1 = np.zeros((9, 2), dtype='float32')
img_pts1[0] = np.array([8, 8], dtype='float32')
img_pts1[1] = np.array([10, 8], dtype='float32')
img_pts1[2] = np.array([12, 8], dtype='float32')
img_pts1[3] = np.array([8, 10], dtype='float32')
img_pts1[4] = np.array([10, 10], dtype='float32')
img_pts1[5] = np.array([12, 10], dtype='float32')
img_pts1[6] = np.array([8, 12], dtype='float32')
img_pts1[7] = np.array([10, 12], dtype='float32')
img_pts1[8] = np.array([12, 12], dtype='float32')

# image points
img_pts2 = np.zeros((9, 2), dtype='float32')
img_pts2[0] = np.array([11.4, 13], dtype='float32')
img_pts2[1] = np.array([16.7, 15.4], dtype='float32')
img_pts2[2] = np.array([24.8, 18.8], dtype='float32')
img_pts2[3] = np.array([11.4, 16.2], dtype='float32')
img_pts2[4] = np.array([16.8, 19.2], dtype='float32')
img_pts2[5] = np.array([24.8, 23.6], dtype='float32')
img_pts2[6] = np.array([11.4, 19.5], dtype='float32')
img_pts2[7] = np.array([16.9, 23], dtype='float32')
img_pts2[8] = np.array([24.8, 28.3], dtype='float32')

ret, calib_mtx1, dist1, calib_mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    [obj_pts], [img_pts1], [img_pts2], imageSize=(30, 30))

print T
print R

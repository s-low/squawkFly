#!/usr/local/bin/python

import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

# This is a test script for extracting the camera parameters with OpenCV
a = np.zeros((100, 100, 100))

# a simulated calibration checkerboard
data = np.zeros((9, 3))
data[0] = [40, 40, 50]
data[1] = [50, 40, 50]
data[2] = [60, 40, 50]
data[3] = [40, 50, 50]
data[4] = [50, 50, 50]
data[5] = [60, 50, 50]
data[6] = [40, 60, 50]
data[7] = [50, 60, 50]
data[8] = [60, 60, 50]

all_x = [row[0] for row in data]
all_y = [row[1] for row in data]
all_z = [row[2] for row in data]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(all_x, all_y, all_z, zdir='z')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

plt.show()

# retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
#     objectPoints=,
#     imagePoints=,
#     imageSize=(,))

# retval,
# cameraMatrix1,
# distCoeffs1,
# cameraMatrix2,
# distCoeffs2,
# R, T, E, F = cv2.stereoCalibrate(objectPoints=, imagePoints1=, imagePoints2=,
#                                  imageSize=(,), cameraMatrix1=, distCoeffs1=,
#                                  cameraMatrix2=, distCoeffs2=,
#                                  criteria=, flags=)

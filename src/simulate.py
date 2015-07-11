#!/usr/local/bin/python

import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')


def setupCameraMatrix(cameraMatrix, focalLength, cx, cy):
    cameraMatrix[0][0] = focalLength
    cameraMatrix[1][1] = focalLength
    cameraMatrix[2][2] = 1
    cameraMatrix[0][2] = cx
    cameraMatrix[1][2] = cy


# object points
objectPoints = np.zeros((9, 3))
objectPoints[0] = [40, 40, 50]
objectPoints[1] = [50, 40, 50]
objectPoints[2] = [60, 40, 50]
objectPoints[3] = [40, 50, 50]
objectPoints[4] = [50, 50, 50]
objectPoints[5] = [60, 50, 50]
objectPoints[6] = [40, 60, 50]
objectPoints[7] = [50, 60, 50]
objectPoints[8] = [60, 60, 50]

# image points given camera intrinsics and extrinsics
rvec = (0, 0, 0)
tvec = (0, 0, 0)
distCoeffs = (0, 0, 0, 3)
cameraMatrix = np.zeros((3, 3))
setupCameraMatrix(cameraMatrix, 50, 50, 50)
print cameraMatrix

imagePoints, jacobian = cv2.projectPoints(
    objectPoints, rvec, tvec, cameraMatrix, distCoeffs)

print imagePoints

# Plotting of the system
a = np.zeros((100, 100, 100))

all_x = [row[0] for row in objectPoints]
all_y = [row[1] for row in objectPoints]
all_z = [row[2] for row in objectPoints]

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

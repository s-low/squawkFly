#!/usr/local/bin/python

import math
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


def plotSimulation(objectPoints):
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


def plotImagePoints(imagePoints):

    all_x = [point[0][0] for point in imagePoints]
    all_y = [point[0][1] for point in imagePoints]

    plt.scatter(all_x, all_y)
    plt.scatter(all_x[0], all_y[0], color='r')
    # plt.axis([0, 200, 0, 200])
    plt.show()


def createRotationX(theta):
    theta = math.radians(theta)
    rmatx = np.zeros((3, 3))
    rmatx[0][0] = 1
    rmatx[1][1] = math.cos(theta)
    rmatx[1][2] = -math.sin(theta)
    rmatx[2][1] = math.sin(theta)
    rmatx[2][2] = math.cos(theta)
    print rmatx
    return rmatx

# object points
obj_pts = np.zeros((9, 3))
obj_pts[0] = [40, 40, 50]
obj_pts[1] = [50, 40, 50]
obj_pts[2] = [60, 40, 50]
obj_pts[3] = [40, 50, 50]
obj_pts[4] = [50, 50, 50]
obj_pts[5] = [60, 50, 50]
obj_pts[6] = [40, 60, 50]
obj_pts[7] = [50, 60, 50]
obj_pts[8] = [60, 60, 50]

# image points given camera intrinsics and extrinsics
# rmat = createRotationX(45)
# rmaty = np.zeros((3, 3))
# rmatz = np.zeros((3, 3))

# r = (r_x.r_y).r_z
# rmat = np.dot(np.dot(rmatx, rmaty), rmatz)
# rvec = cv2.Rodrigues(rmat)

rvec = (0.5, 0.5, 0)  # rotation relative to the frame
tvec = (0, 0, 0)  # translation relative to the frame
distCoeffs = (0, 0, 0, 0)
cameraMatrix = np.zeros((3, 3))
setupCameraMatrix(cameraMatrix, 50, 0, 0)

imagePoints, jacobian = cv2.projectPoints(
    obj_pts, rvec, tvec, cameraMatrix, distCoeffs)

# print imagePoints
plotImagePoints(imagePoints)

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

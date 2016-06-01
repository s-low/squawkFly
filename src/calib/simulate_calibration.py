#!/usr/local/bin/python

''' simulate_calibration.py

    Test script to simulate the calibration process using Zhang's method
    as implemented in OpenCV. Completely academic exercise to help understand
    what's going on here.
'''


import math
import cv2
import cv2.cv as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')


def setupCameraMatrix(calibMatrix, focalLength, cx, cy):
    calibMatrix[0][0] = focalLength
    calibMatrix[1][1] = focalLength
    calibMatrix[2][2] = 1
    calibMatrix[0][2] = cx
    calibMatrix[1][2] = cy


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

# camera matrix
calibMatrix = np.zeros((3, 3))
focalLength = 10
cx = 0
cy = 0
setupCameraMatrix(calibMatrix, focalLength, 0, 0)

# projections into image planes with the camera in different poses
distCoeffs = (0, 0, 0, 0)

tvec = (0, 0, 50)
rvec = (0, 0, 0)
img_pts1, jacobian = cv2.projectPoints(
    obj_pts, rvec, tvec, calibMatrix, distCoeffs)

tvec = (0, 0, 50)
rvec = (0, 0.5, 0)
img_pts2, jacobian = cv2.projectPoints(
    obj_pts, rvec, tvec, calibMatrix, distCoeffs)

tvec = (0, 0, 100)
rvec = (0, 0, 0)
img_pts3, jacobian = cv2.projectPoints(
    obj_pts, rvec, tvec, calibMatrix, distCoeffs)

tvec = (0, 0, 30)
rvec = (0.5, 0.5, 0)
img_pts4, jacobian = cv2.projectPoints(
    obj_pts, rvec, tvec, calibMatrix, distCoeffs)

tvec = (10, 0, 90)
rvec = (1, 0.5, 1)
img_pts5, jacobian = cv2.projectPoints(
    obj_pts, rvec, tvec, calibMatrix, distCoeffs)

tvec = (5, 0, 50)
rvec = (0.5, 1, 1)
img_pts6, jacobian = cv2.projectPoints(
    obj_pts, rvec, tvec, calibMatrix, distCoeffs)

tvec = (0, 0, 40)
rvec = (0.1, 0.1, 0.1)
img_pts7, jacobian = cv2.projectPoints(
    obj_pts, rvec, tvec, calibMatrix, distCoeffs)

# plot resulting imagePoints
plotSimulation(obj_pts)
plotImagePoints(img_pts1)
plotImagePoints(img_pts2)
plotImagePoints(img_pts3)
plotImagePoints(img_pts4)
plotImagePoints(img_pts5)
plotImagePoints(img_pts6)
plotImagePoints(img_pts7)

# calibrate camera expects a list of arrays such as these
obj_pts_list = [obj_pts, obj_pts, obj_pts, obj_pts, obj_pts, obj_pts]
img_pts_list = [img_pts1, img_pts2, img_pts3, img_pts4, img_pts5, img_pts6]

ret, calib_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts_list, img_pts_list, (30, 30), None, None)

print "> calib_matrix:\n", calib_mtx

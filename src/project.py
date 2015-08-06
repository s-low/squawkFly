#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import structureTools as tools

np.set_printoptions(suppress=True)
plt.style.use('ggplot')


def main():
    folder = sys.argv[1]
    data_3d = getData(folder)

    data_3d = np.array(data_3d, dtype='float32')

    # artificial camera with f=5 units, cx = cy = 5 units
    K = tools.CalibArray(1000, 640, 360)
    dist = (0, 0, 0, 0)

    rt2 = math.sqrt(2)

    # 45 cc about z
    z45cc = np.mat([[1 / rt2, -1 / rt2, 0],
                    [1 / rt2, 1 / rt2, 0],
                    [0, 0, 1]], dtype='float32')

    z90cc = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype='float32')
    z90cw = np.mat([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype='float32')

    z180 = np.mat([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype='float32')
    x180 = np.mat([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype='float32')
    y180 = np.mat([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype='float32')

    x90cw = np.mat([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype='float32')
    x90cc = np.mat([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype='float32')
    y90cc = np.mat([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype='float32')

    z90cc_vec, jacobian = cv2.Rodrigues(z90cc)
    z90cc_vec, jacobian = cv2.Rodrigues(z90cc)
    y90cc_vec, jacobian = cv2.Rodrigues(y90cc)
    x90cc_vec, jacobian = cv2.Rodrigues(x90cc)

    # projections into image planes with the camera in different poses
    tvec1 = (0, 0, 1000)
    rvec1 = (0, 0, 0)

    tvec2 = (0, 500, 500)
    rvec2 = (0, 0, 0)

    img_pts1 = project(data_3d, K, y180, tvec1)
    img_pts2 = project(data_3d, K, x90cc, tvec2)

    # img_pts1, jacobian = cv2.projectPoints(
    #     data_3d, rvec1, tvec1, K, dist)

    # img_pts2, jacobian = cv2.projectPoints(
    #     data_3d, rvec2, tvec2, K, dist)

    img_pts1 = np.reshape(img_pts1, (len(img_pts1), 2, 1))
    img_pts2 = np.reshape(img_pts2, (len(img_pts2), 2, 1))

    plotSimulation(data_3d)
    plotImagePoints(img_pts1)
    plotImagePoints(img_pts2)

    writeData(folder, img_pts1, img_pts2)


# own implementation of cv2.projectPoints() - project 3d into 2d plane
# given intrinsic, rotation matrix and translation vector
def project(objectPoints, K, R, t):

    print "--------- MANUAL PROJECTION -----------"
    objectPoints = cv2.convertPointsToHomogeneous(objectPoints)
    objectPoints = fixExtraneousParentheses(objectPoints)
    print objectPoints

    imagePoints = []
    print "K:\n", K

    t = np.mat(t)
    t = t.T
    print "R:\n", R
    print "t:\n", t

    Rt = np.concatenate((R, t), 1)
    print "R|t:\n", Rt

    P = K * Rt
    print "P = k(R|t):\n", P

    for X in objectPoints:
        x = np.mat(X).T
        x_ = P * x
        imagePoints.append(x_)

    # image points are homogeneous (xz, yz, z) -> (x, y, 1)
    normed = []
    for p in imagePoints:
        p = p / p[2]
        normed.append(p)
    normed = np.array(normed)
    normed = np.delete(normed, 2, 1)
    return normed


def getData(folder):
    path = 'tests/' + str(folder) + '/'
    pts1 = []
    pts2 = []
    original_3Ddata = []

    with open(path + '3d.txt') as datafile:
        data = datafile.read()
        datafile.close()

    data = data.split('\n')
    for row in data:
        x = float(row.split()[0])
        y = float(row.split()[1])
        z = float(row.split()[2])
        original_3Ddata.append([x, y, z])

    return original_3Ddata


def writeData(folder, pts1, pts2):
    path = 'tests/' + str(folder) + '/'

    startoffile = True
    outfile = open(path + 'pts1.txt', 'w')

    for p in pts1:
        dstring = str(p[0, 0]) + ' ' + str(p[1, 0])
        if not startoffile:
            outfile.write('\n')
        outfile.write(dstring)
        startoffile = False
    outfile.close()

    startoffile = True
    outfile = open(path + 'pts2.txt', 'w')

    for p in pts2:
        dstring = str(p[0, 0]) + ' ' + str(p[1, 0])
        if not startoffile:
            outfile.write('\n')
        outfile.write(dstring)
        startoffile = False
    outfile.close()


def fixExtraneousParentheses(points):
    temp = []
    for p in points:
        p = p[0]
        temp.append(p)

    new = np.mat(temp)
    return new


# Plotting of the system in 3d
def plotSimulation(objectPoints):

    all_x = [row[0] for row in objectPoints]
    all_y = [row[1] for row in objectPoints]
    all_z = [row[2] for row in objectPoints]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(all_x, all_y, all_z, zdir='z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 100)
    # ax.set_zlim(0, 100)

    plt.show()


def plotImagePoints(imagePoints):
    all_x = []
    all_y = []
    all_x = [point[0] for point in imagePoints]
    all_y = [point[1] for point in imagePoints]

    plt.scatter(all_x, all_y)
    plt.scatter(all_x[0], all_y[0], color='r')
    plt.show()

main()

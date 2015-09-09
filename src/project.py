#!/usr/local/bin/python

''' project.py <3d.txt>

    create artificial image data of a 3d model, and save that image data
    to the same directory as the model resides in.

'''


import sys
import cv2
import cv2.cv as cv
import numpy as np
import numpy.random as random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import structureTools as tools
import os.path
import plotting as plot

np.set_printoptions(suppress=True)


def main():

    path = sys.argv[1]
    folder = os.path.dirname(path)

    data_3d = getData(path)

    data_3d = np.array(data_3d, dtype='float32')

    # Artificial cameras - both have the same intrinsics
    K = tools.CalibArray(1000, 640, -360)
    dist = (0, 0, 0, 0)

    # Some numbers we use a lot
    rt2 = math.sqrt(2)
    rt2on2 = rt2 / 2

    # Lots of rotation matrices...
    # naming: z45cc = 45deg counter-clockwise rotation about z

    z45cc = np.mat([[1 / rt2, -1 / rt2, 0],
                    [1 / rt2, 1 / rt2, 0],
                    [0, 0, 1]], dtype='float32')

    nothing = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float32')

    z90cc = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype='float32')
    z90cw = np.mat([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype='float32')

    z180 = np.mat([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype='float32')
    x180 = np.mat([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype='float32')
    y180 = np.mat([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype='float32')

    x90cw = np.mat([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype='float32')
    x90cc = np.mat([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype='float32')

    y90cc = np.mat([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype='float32')
    y90cw = np.mat([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype='float32')

    z90cc_vec, jacobian = cv2.Rodrigues(z90cc)
    z90cc_vec, jacobian = cv2.Rodrigues(z90cc)
    y90cc_vec, jacobian = cv2.Rodrigues(y90cc)
    x90cc_vec, jacobian = cv2.Rodrigues(x90cc)

    y45cc = np.mat([[rt2on2, 0, rt2on2],
                    [0, 1, 0],
                    [-rt2on2, 0, rt2on2]], dtype='float32')

    y45cw = np.mat([[rt2on2, 0, -rt2on2],
                    [0, 1, 0],
                    [rt2on2, 0, rt2on2]], dtype='float32')

    # cameras have different poses
    tvec1 = (-5, 1, 14)
    tvec2 = (5, 1, 13)

    # Project the model into each camera
    img_pts1 = project(data_3d, K, y45cc, tvec1)
    img_pts2 = project(data_3d, K, y45cw, tvec2)

    img_pts1 = np.reshape(img_pts1, (len(img_pts1), 2, 1))
    img_pts2 = np.reshape(img_pts2, (len(img_pts2), 2, 1))

    # Plot the model and images
    plot.plot3D(data_3d)
    plotImagePoints(img_pts1)
    plotImagePoints(img_pts2)

    # Add noise to the image data if desired
    try:
        noise = sys.argv[2]
        img_pts1 = addNoise(float(noise), img_pts1)
        img_pts2 = addNoise(float(noise), img_pts2)
    except indexError:
        continue

    writeData(folder, img_pts1, img_pts2)


# own implementation of cv2.projectPoints() - project 3d into 2d plane
# given intrinsic, rotation matrix and translation vector
def project(objectPoints, K, R, t):

    print "--------- MANUAL PROJECTION -----------"
    objectPoints = cv2.convertPointsToHomogeneous(objectPoints)
    objectPoints = fixExtraneousParentheses(objectPoints)

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


def getData(path):
    pts1 = []
    pts2 = []
    original_3Ddata = []

    folder = os.path.dirname(path)

    with open(path) as datafile:
        data = datafile.read()
        datafile.close()

    data = data.split('\n')

    # get rid of any empty line at the end of file
    if data[-1] in ['\n', '\r\n', '']:
        data.pop(-1)

    for row in data:
        x = float(row.split()[0])
        y = float(row.split()[1])
        z = float(row.split()[2])
        original_3Ddata.append([x, y, z])

    return original_3Ddata


def writeData(folder, pts1, pts2):
    startoffile = True
    outfile = open(os.path.join(folder, 'pts1.txt'), 'w')

    for p in pts1:
        dstring = str(p[0, 0]) + ' ' + str(p[1, 0])
        if not startoffile:
            outfile.write('\n')
        outfile.write(dstring)
        startoffile = False
    outfile.close()

    startoffile = True
    outfile = open(os.path.join(folder, 'pts2.txt'), 'w')

    for p in pts2:
        dstring = str(p[0, 0]) + ' ' + str(p[1, 0])
        if not startoffile:
            outfile.write('\n')
        outfile.write(dstring)
        startoffile = False
    outfile.close()


# little hack to fix some weird parantheses conventions returned by numpy
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

    plt.show()


def plotImagePoints(imagePoints):
    all_x = []
    all_y = []
    all_x = [point[0] for point in imagePoints]
    all_y = [point[1] for point in imagePoints]

    plt.scatter(all_x, all_y)
    plt.scatter(all_x[0], all_y[0], color='r')
    plt.xlim((0, 1280))
    plt.ylim((-720, 0))
    plt.xlabel('Graphical X (px)')
    plt.ylabel('Graphical Y (px)')
    plt.show()


# add some random noise to n image point set
def addNoise(sigma, points):
    new = []
    for p in points:

        nx = random.normal(0, sigma)
        ny = random.normal(0, sigma)

        n0 = p[0] + nx
        n1 = p[1] + ny
        n = [n0, n1]
        new.append(n)

    return np.array(new, dtype='float32')


main()

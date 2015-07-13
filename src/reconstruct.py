#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

Point = namedtuple("Point", "x y")


def run():
    # 1 = RHS d5000
    pts1_raw = [[726, 267],
                [761, 266],
                [715, 463],
                [749, 461],
                [667, 432],
                [677, 256],
                [971, 465],
                [448, 383],
                [1035, 531],
                [572, 165]]

    # 2 = LHS G3
    pts2_raw = [[664, 391],
                [684, 379],
                [660, 587],
                [685, 571],
                [554, 554],
                [550, 365],
                [924, 536],
                [229, 554],
                [1179, 646],
                [342, 260]]

    pts1 = np.array(pts1_raw, dtype='float32')
    pts2 = np.array(pts2_raw, dtype='float32')

    # Given camera CALIBRATION MATRICES
    K1 = np.mat(CalibArray(993, 655, 371))  # d5000
    K2 = np.mat(CalibArray(1091, 644, 412))  # g3

    # Find FUNDAMENTAL MATRIX from point correspondences
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv.CV_FM_8POINT)
    print "Fundamental:\n", F

    testFundamental(F, pts1, pts2)
    sys.exit()

    # turn it into the ESSENTIAL MATRIX using calibration matrices K1/K2
    # where 1 corresponds to the camera with P1 = [I|0]
    E = K2.T * np.mat(F) * K1
    print "Essential:\n", E

    # Find CAMERA MATRICES from E
    # where one of them has P1 = [I|0] get the second camera matrix P2 = [R|t]
    # this relies on Hartley and Zisserman sec 9.6

    W, W_inv = initSVDarrays()

    s, u, vt = cv2.SVDecomp(E)

    # HZ 9.19 (may need to check all four)
    R1 = np.mat(u) * np.mat(W) * np.mat(vt)
    R2 = np.mat(u) * np.mat(W.T) * np.mat(vt)
    t1 = u[:, 2]      # translation is third col of U
    t2 = -1 * u[:, 2]

    print "R:\n", R1, R2
    print "t:\n", t1, t2

    R = R2
    t = t2
    P1 = BoringCameraArray()
    P2 = CameraArray(R, t)

    print "P1:\n", P1
    print "P2:\n", P2

    # triangulate the points (don't use cv2.triangulatePoints)
    points3d = []

    for i in range(0, len(pts1_raw)):
        x1 = pts1_raw[i][0]
        y1 = pts1_raw[i][0]
        x2 = pts2_raw[i][0]
        y2 = pts2_raw[i][0]

        u1 = Point(x1, y1)
        u2 = Point(x2, y2)

        X = LinearTriangulation(P1, u1, P2, u2)
        points3d.append(X[1])

    plot3D(points3d)


def testFundamental(F, pts1, pts2):
    # check that xFx = 0
    pts1 = cv2.convertPointsToHomogeneous(src=pts1)
    pts2 = cv2.convertPointsToHomogeneous(src=pts2)
    err = 0
    for i in range(0, len(pts1)):
        err += np.mat(pts1[i]) * np.mat(F) * np.mat(pts2[i].T)

    print "avg px error in xFx:", err / len(pts1)


def CalibArray(focalLength, cx, cy):
    calibArray = np.zeros((3, 3), dtype='float32')
    calibArray[0][0] = focalLength
    calibArray[1][1] = focalLength
    calibArray[2][2] = 1
    calibArray[0][2] = cx
    calibArray[1][2] = cy
    return calibArray


def initSVDarrays():
    W = np.zeros((3, 3), dtype='float32')
    W_inv = np.zeros((3, 3), dtype='float32')

    W[0][0] = 0  # HZ 9.13
    W[0][1] = -1
    W[0][2] = 0
    W[1][0] = 1
    W[1][1] = 0
    W[1][2] = 0
    W[2][0] = 0
    W[2][1] = 0
    W[2][2] = 1

    W_inv[0][0] = 0  # HZ 9.13
    W_inv[0][1] = 1
    W_inv[0][2] = 0
    W_inv[1][0] = -1
    W_inv[1][1] = 0
    W_inv[1][2] = 0
    W_inv[2][0] = 0
    W_inv[2][1] = 0
    W_inv[2][2] = 1
    return W, W_inv


def BoringCameraArray():
    P = np.zeros((3, 4), dtype='float32')
    P[0][0] = 1
    P[1][1] = 1
    P[2][2] = 1
    return P


def CameraArray(R, t):
    P = np.zeros((3, 4), dtype='float32')
    P[0][0] = R[0, 0]
    P[0][1] = R[0, 1]
    P[0][2] = R[0, 2]
    P[0][3] = t[0]

    P[1][0] = R[1, 0]
    P[1][1] = R[1, 1]
    P[1][2] = R[1, 2]
    P[1][3] = t[1]

    P[2][0] = R[2, 0]
    P[2][1] = R[2, 1]
    P[2][2] = R[2, 2]
    P[2][3] = t[2]

    return P


def plot3D(objectPoints):
    # Plotting of the system
    print "PLOT"
    for point in objectPoints:
        print point[0], point[1], point[2]

    all_x = [point[0] for point in objectPoints]
    all_y = [point[1] for point in objectPoints]
    all_z = [point[2] for point in objectPoints]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(all_x, all_y, all_z, zdir='z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def LinearTriangulation(P1, u1, P2, u2):

    A = np.zeros((4, 3), dtype='float32')
    B = np.zeros((4, 1), dtype='float32')
    X = np.zeros((3, 1), dtype='float32')

    A[0][0] = u1.x * P1[2][0] - P1[0][0]
    A[0][1] = u1.x * P1[2][1] - P1[0][1]
    A[0][2] = u1.x * P1[2][2] - P1[0][2]

    A[1][0] = u1.y * P1[2][0] - P1[1][0]
    A[1][1] = u1.y * P1[2][1] - P1[1][1]
    A[1][2] = u1.y * P1[2][2] - P1[1][2]

    A[2][0] = u2.x * P2[2][0] - P2[0][0]
    A[2][1] = u2.x * P2[2][1] - P2[0][1]
    A[2][2] = u2.x * P2[2][2] - P2[0][2]

    A[3][0] = u2.y * P2[2][0] - P2[1][0]
    A[3][1] = u2.y * P2[2][1] - P2[1][1]
    A[3][2] = u2.y * P2[2][2] - P2[1][2]

    # print "A:\n", A

    B[0][0] = -(u1.x * P1[2][3] - P1[0][3])
    B[0][0] = -(u1.y * P1[2][3] - P1[1][3])
    B[0][0] = -(u2.x * P2[2][3] - P2[0][3])
    B[3][0] = -(u2.y * P2[2][3] - P2[1][3])

    # print "B:\n", B
    X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return X


def IterativeLinearTriangulation(P1, u1, P2, u2):

    A = np.zeros((4, 3), dtype='float32')
    B = np.zeros((4, 1), dtype='float32')
    X = np.zeros((3, 1), dtype='float32')

    A[0][0] = u1.x * P1[2][0] - P1[0][0]
    A[0][1] = u1.x * P1[2][1] - P1[0][1]
    A[0][2] = u1.x * P1[2][2] - P1[0][2]

    A[1][0] = u1.y * P1[2][0] - P1[1][0]
    A[1][1] = u1.y * P1[2][1] - P1[1][1]
    A[1][2] = u1.y * P1[2][2] - P1[1][2]

    A[2][0] = u2.x * P2[2][0] - P2[0][0]
    A[2][1] = u2.x * P2[2][1] - P2[0][1]
    A[2][2] = u2.x * P2[2][2] - P2[0][2]

    A[3][0] = u2.y * P2[2][0] - P2[1][0]
    A[3][1] = u2.y * P2[2][1] - P2[1][1]
    A[3][2] = u2.y * P2[2][2] - P2[1][2]

    # print "A:\n", A

    B[0][0] = -(u1.x * P1[2][3] - P1[0][3])
    B[0][0] = -(u1.y * P1[2][3] - P1[1][3])
    B[0][0] = -(u2.x * P2[2][3] - P2[0][3])
    B[3][0] = -(u2.y * P2[2][3] - P2[1][3])

    # print "B:\n", B
    X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return X


run()

#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np
import math
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

Point = namedtuple("Point", "x y")
np.set_printoptions(suppress=True)


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
    K1 = np.mat(CalibArray(993, 640, 360))  # d5000
    K2 = np.mat(CalibArray(1091, 640, 360))  # g3

    # Find FUNDAMENTAL MATRIX from point correspondences
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv.CV_FM_8POINT, 90)
    print "\n> Fundamental:\n", F
    testFundamentalReln(F, pts1, pts2)

    # transform point correspondences to normalized image coordinates
    # apply the inverse of the calibration matrices to them
    npts1 = normalise_homogenise(pts1, K1)
    npts2 = normalise_homogenise(pts2, K2)

    # compute ESSENTIAL MATRIX from F, K1, K2 (HZ 9.12)
    E = K2.T * np.mat(F) * K1
    print "\n> Essential:\n", E
    testEssentialReln(E, npts1, npts2)

    # Find CAMERA MATRICES from E (HZ 9.6.2)
    # P1 = [I|0] and P2 = [R|t]
    w, u, vt = cv2.SVDecomp(E)
    print "\n> singular values:\n", w

    # a matrix is only essential if two of it's singular values are equal
    # and the third is zero
    diag = np.mat(np.diag([1, 1, 0]))
    E_new = np.mat(u) * diag * np.mat(vt)
    print "\n> E_new = u * diag(1,1,0) * vt:\n", E_new
    testEssentialReln(E_new, npts1, npts2)

    # decompose the second, strictly essential matrix (very unsure about this)
    w2, u2, vt2 = cv2.SVDecomp(E_new)
    print "\n> singular values:\n", w2

    # HZ 9.19 (need to check all four pairings):
    W, W_inv = initWarrays()

    R1 = np.mat(u2) * np.mat(W) * np.mat(vt2)
    t1 = u2[:, 2]
    # inFrontOfBothCameras(pts1, pts2, R1, t1)

    R2 = np.mat(u2) * np.mat(W.T) * np.mat(vt2)
    t1 = u2[:, 2]
    t2 = -1 * u2[:, 2]

    print "\n> R1:\n", R1
    print "> R2:\n", R2
    print "\n> t1:\n", t1
    print "> t2:\n", t2

    # enforce positive depth combination of Rt
    if testRtCombo(R1, t1, pts1_raw, pts2_raw):
        R = R1
        t = t1

    elif testRtCombo(R1, t2, pts1_raw, pts2_raw):
        R = R1
        t = t2

    elif testRtCombo(R2, t1, pts1_raw, pts2_raw):
        R = R1
        t = t2

    elif testRtCombo(R2, t2, pts1_raw, pts2_raw):
        R = R1
        t = t2

    # CAMERA MATRICES
    P1 = BoringCameraArray()
    P2 = CameraArray(R, t)

    print "\n> P1:\n", P1
    print "\n> P2:\n", P2

    # TRIANGULATION
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
    reprojectionError(K1, P1, K2, P2, pts1, pts2, points3d)


def testFundamentalReln(F, pts1, pts2):
    # check that xFx = 0

    pts1 = cv2.convertPointsToHomogeneous(pts1)
    pts2 = cv2.convertPointsToHomogeneous(pts2)
    err = 0
    for i in range(0, len(pts1)):
        err += np.mat(pts1[i]) * np.mat(F) * np.mat(pts2[i]).T

    print "> avg px error in xFx:", err[0, 0] / len(pts1)


def testEssentialReln(E, nh_pts1, nh_pts2):
    # check that x'Ex = 0 for normalised coordinates x, x'
    err = 0
    m = np.mat(nh_pts1[0])
    E = np.mat(E)
    m1 = np.mat(nh_pts2[0]).T

    for i in range(0, len(nh_pts1)):
        err += np.mat(nh_pts1[i]) * E * np.mat(nh_pts2[i]).T

    print "> avg normalised px error in x'Ex:", err[0, 0] / len(nh_pts1)


# convert a set of (x, y) image points to normalised, homogenous image coords
def normalise_homogenise(pts, K):
    np.set_printoptions(suppress=True)
    pts = cv2.convertPointsToHomogeneous(pts)
    K_inv = np.linalg.inv(K)

    npts = np.zeros((len(pts), 3))
    for i, x in enumerate(pts):
        xn = K_inv * x.T
        npts[i][0] = xn[0]
        npts[i][1] = xn[1]
        npts[i][2] = xn[2]
    return npts


def testRtCombo(R, t, pts1_raw, pts2_raw):
    P1 = BoringCameraArray()
    P2 = CameraArray(R, t)
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

    for point in points3d:
        if point[2] < 0:
            return False

    return True


def inFrontOfBothCameras(pts1, pts2, R, t):
    print "\n> testing Rt combination:"
    pts1 = cv2.convertPointsToHomogeneous(pts2)
    pts2 = cv2.convertPointsToHomogeneous(pts2)

    for first, second in zip(pts1, pts2):
        print second.shape, t.shape

        A = (np.mat(R[0, :]) - (second[0] * np.mat(R[2, :]))) * np.mat(t)
        B = np.dot(R[0, :] - second[0] * R[2, :], second)

        first_z = A / B

        first_3d_point = np.array(
            [first[0] * first_z, second[0] * first_z, first_z])

        second_3d_point = np.dot(R.T, first_3d_point) - np.dot(R.T, t)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False


# used for checking the triangulation
def reprojectionError(K1, P1, K2, P2, pts1, pts2, points3d):

    new = np.zeros((len(points3d), 4))
    for i, point in enumerate(points3d):
        new[i][0] = point[0]
        new[i][1] = point[1]
        new[i][2] = point[2]
        new[i][3] = 1
    total = 0

    # for each 3d point
    for i, X in enumerate(new):
        # project into each camera with retrieved P matrices
        xp1 = P1 * np.mat(X).T
        xp2 = P2 * np.mat(X).T

        # and get the orginally measured points
        x1 = pts1[i]
        x2 = pts2[i]

        # difference between them is:
        dist1 = math.hypot(xp1[0] - x1[0], xp1[1] - x1[1])
        dist2 = math.hypot(xp2[0] - x2[0], xp2[1] - x2[1])

        total += dist1 + dist2

    print "\n> avg reprojection error in triangulation:", \
        total / (2 * len(points3d))


def CalibArray(focalLength, cx, cy):
    calibArray = np.zeros((3, 3), dtype='float32')
    calibArray[0][0] = focalLength
    calibArray[1][1] = focalLength
    calibArray[2][2] = 1
    calibArray[0][2] = cx
    calibArray[1][2] = cy
    return calibArray


def initWarrays():
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
    print "\n> plotting data:"
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

    B[0][0] = -(u1.x * P1[2][3] - P1[0][3])
    B[0][0] = -(u1.y * P1[2][3] - P1[1][3])
    B[0][0] = -(u2.x * P2[2][3] - P2[0][3])
    B[3][0] = -(u2.y * P2[2][3] - P2[1][3])

    X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return X


def IterativeLinearTriangulation(P1, u1, P2, u2):
    EPSILON = 5

    # weightings
    wi1 = 1
    wi2 = 1

    # systen matrices
    A = np.zeros((4, 3), dtype='float32')
    B = np.zeros((4, 1), dtype='float32')
    X = np.zeros((3, 1), dtype='float32')

    # 3d homogeneous coord (x,y,z,1)
    X = np.zeros((4, 1), dtype='float32')

    for i in range(0, 10):
        # compute the linear triangulation to (x,y,z) as normal
        X_ = LinearTriangulation(P1, u1, P2, u2)[1]
        X[0] = X_[0]
        X[1] = X_[1]
        X[2] = X_[2]
        X[3] = 1

        # calculate weightings
        p2x1 = np.mat(P1)[2] * X
        p2x2 = np.mat(P2)[2] * X

        if abs(wi1 - p2x1) <= EPSILON and abs(wi2 - p2x2) <= EPSILON:
            break

        wi1 = p2x1
        wi2 = p2x2

        # reweight equations and solve
        A[0][0] = (u1.x * P1[2][0] - P1[0][0]) / wi1
        A[0][1] = (u1.x * P1[2][1] - P1[0][1]) / wi1
        A[0][2] = (u1.x * P1[2][2] - P1[0][2]) / wi1

        A[1][0] = (u1.y * P1[2][0] - P1[1][0]) / wi1
        A[1][1] = (u1.y * P1[2][1] - P1[1][1]) / wi1
        A[1][2] = (u1.y * P1[2][2] - P1[1][2]) / wi1

        A[2][0] = (u2.x * P2[2][0] - P2[0][0]) / wi2
        A[2][1] = (u2.x * P2[2][1] - P2[0][1]) / wi2
        A[2][2] = (u2.x * P2[2][2] - P2[0][2]) / wi2

        A[3][0] = (u2.y * P2[2][0] - P2[1][0]) / wi2
        A[3][1] = (u2.y * P2[2][1] - P2[1][1]) / wi2
        A[3][2] = (u2.y * P2[2][2] - P2[1][2]) / wi2

        B[0][0] = -(u1.x * P1[2][3] - P1[0][3])
        B[0][0] = -(u1.y * P1[2][3] - P1[1][3])
        B[0][0] = -(u2.x * P2[2][3] - P2[0][3])
        B[3][0] = -(u2.y * P2[2][3] - P2[1][3])

        X_ = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
        X[0] = X_[0]
        X[1] = X_[1]
        X[2] = X_[2]
        X[3] = 1

    return X

run()

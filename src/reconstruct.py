#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D

import triangulation as tri
import structureTools as struc

np.set_printoptions(suppress=True)
plt.style.use('ggplot')

Point = namedtuple("Point", "x y")

# Calibration matrices:
K1 = np.mat(struc.CalibArray(5, 5, 5))
K2 = np.mat(struc.CalibArray(5, 5, 5))

# simulated projection data with project.py

# data_3d = [[0, 0, 0],
#                [10, 10, 10],
#                [20, 20, 20],
#                [30, 30, 30],
#                [40, 40, 40],
#                [50, 50, 50],
#                [60, 40, 40],
#                [70, 30, 30],
#                [80, 20, 20],
#                [90, 10, 10],
#                [100, 0, 0]]

pts1_raw = [[5.0, 5.0],
            [6.25, 6.25],
            [7.0, 7.0],
            [7.5, 7.5],
            [7.85, 7.85714293],
            [8.12, 8.125],
            [9.28, 7.85714293],
            [10.83, 7.5],
            [13.00, 7.0],
            [16.25, 6.25],
            [21.66, 5.00]]

pts2_raw = [[5.0, 6.875],
            [6.23262691, 7.61429834],
            [7.11896372, 8.14590073],
            [7.78696489, 8.54655075],
            [8.30845928, 8.85933018],
            [8.72688103, 9.11028957],
            [9.9604845, 9.08724403],
            [11.71813202, 9.05440903],
            [14.42380333, 9.00386333],
            [19.12773895, 8.91598797],
            [29.33972359, 8.72521496]]

# Image coords: (x, y)
pts1 = np.array(pts1_raw, dtype='float32')
pts2 = np.array(pts2_raw, dtype='float32')

# Normalised homogenous image coords: (x, y, 1)
norm_pts1 = struc.normalise_homogenise(pts1, K1)
norm_pts2 = struc.normalise_homogenise(pts2, K2)

inhomog_norm_pts1 = np.delete(norm_pts1, 2, 1)
inhomog_norm_pts2 = np.delete(norm_pts2, 2, 1)

# Arrays FOR Rt computation
W, W_inv = struc.initWarrays()  # HZ 9.13


def run():
    # FUNDAMENTAL MATRIX
    F = getFundamentalMatrix(pts1, pts2)

    # ESSENTIAL MATRIX (HZ 9.12)
    E, w, u, vt = getEssentialMatrix(F, K1, K2)

    # CONSTRAINED ESSENTIAL MATRIX
    E_prime, w2, u2, vt2 = getConstrainedEssentialMatrix(u, vt)

    # scale = E[0, 0] / E_prime[0, 0]
    # E_prime = E_prime * scale
    # print "\n> scaled:\n", E_prime

    # PROJECTION/CAMERA MATRICES from E (or E_prime?) (HZ 9.6.2)
    P1, P2 = getNormalisedPMatrices(E, u, vt)
    P1_mat = np.mat(P1)
    P2_mat = np.mat(P2)

    # FULL PROJECTION MATRICES (with K) P = K[Rt]
    KP1 = K1 * P1_mat
    KP2 = K2 * P2_mat

    print "\n> KP1:\n", KP1
    print "\n> KP2:\n", KP2

    # TRIANGULATION
    points3d = triangulateLS(KP1, KP2, pts1, pts2)
    # points4d = triangulateCV(KP1, KP2, pts1, pts2)

    # PLOTTING
    plot3D(points3d)
    reprojectionError(K1, P1_mat, K2, P2_mat, pts1, pts2, points3d)


# get the Fundamental matrix by the normalised eight point algorithm
def getFundamentalMatrix(pts1, pts2):

    # raw 8-point algorithm
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv.CV_FM_8POINT)

    print "\n> Fundamental:\n", F
    testFundamentalReln(F, pts1, pts2)
    return F


def getEssentialMatrix(F, K1, K2):
    E = K2.T * np.mat(F) * K1
    print "\n> Essential:\n", E
    testEssentialReln(E, norm_pts1, norm_pts2)

    w, u, vt = cv2.SVDecomp(E)
    print "\n> Singular values:\n", w
    return E, w, u, vt


# https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_3:_Enforcing_the_internal_constraint
def getConstrainedEssentialMatrix(u, vt):
    diag = np.mat(np.diag([1, 1, 0]))

    E_prime = np.mat(u) * diag * np.mat(vt)
    print "\n> Constrained Essential = u * diag(1,1,0) * vt:\n", E_prime
    testEssentialReln(E_prime, norm_pts1, norm_pts2)

    w2, u2, vt2 = cv2.SVDecomp(E_prime)
    print "\n> Singular values:\n", w2

    return E_prime, w2, u2, vt2


def getNormalisedPMatrices(E, u, vt):
    R1 = np.mat(u) * np.mat(W) * np.mat(vt)
    R2 = np.mat(u) * np.mat(W.T) * np.mat(vt)
    t1 = u[:, 2]
    t2 = -1 * u[:, 2]

    R, t = getValidRtCombo(R1, R2, t1, t2)

    # NORMALISED CAMERA MATRICES P = [Rt]
    P1 = BoringCameraArray()  # I|0
    P2 = CameraArray(R, t)    # Rt

    print "\n> P1:\n", P1
    print "\n> P2:\n", P2

    return P1, P2


def getValidRtCombo(R1, R2, t1, t2):
    # enforce positive depth combination of Rt using normalised coords
    if testRtCombo(R1, t1, norm_pts1, norm_pts2):
        print "\n> RT: R1 t1"
        R = R1
        t = t1

    elif testRtCombo(R1, t2, norm_pts1, norm_pts2):
        print "\n> RT: R1 t2"
        R = R1
        t = t2

    elif testRtCombo(R2, t1, norm_pts1, norm_pts2):
        print "\n> RT: R2 t1"
        R = R2
        t = t1

    elif testRtCombo(R2, t2, norm_pts1, norm_pts2):
        print "\n> RT: R2 t2"
        R = R2
        t = t2

    else:
        print "ERROR: No positive depth Rt combination"
        sys.exit()

    print "R:\n", R
    print "t:\n", t
    return R, t


def triangulateLS(P1, P2, pts1, pts2):
    points3d = []

    for i in range(0, len(pts1)):

        x1 = pts1[i][0]
        y1 = pts1[i][1]

        x2 = pts2[i][0]
        y2 = pts2[i][1]

        p1 = Point(x1, y1)
        p2 = Point(x2, y2)

        X = tri.LinearTriangulation(P1, p1, P2, p2)

        points3d.append(X[1])

    return points3d


# expects normalised points
def triangulateCV(KP1, KP2, pts1, pts2):
    points4d = cv2.triangulatePoints(KP1, KP2, pts1.T, pts2.T)
    points4d = points4d.T
    print "\n> cv2.triangulatePoints:\n"
    for point in points4d:
        k = 1 / point[3]
        point = point * k

    return points4d


def testFundamentalReln(F, pts1, pts2):
    # check that xFx = 0 for homog coords x x'
    F = np.mat(F)
    is_singular(F)

    pts1_hom = cv2.convertPointsToHomogeneous(pts1)
    pts2_hom = cv2.convertPointsToHomogeneous(pts2)

    err = 0
    for i in range(0, len(pts1_hom)):
        err += abs(np.mat(pts1_hom[i]) * F * np.mat(pts2_hom[i]).T)

    err = err[0, 0] / len(pts1_hom)
    print "> avg error in x'Fx:", err

    # test the epilines
    pts1_epi = pts1.reshape(-1, 1, 2)
    pts2_epi = pts2.reshape(-1, 1, 2)

    # lines computed from pts1
    lines1 = cv2.computeCorrespondEpilines(pts1_epi, 2, F)
    lines1 = lines1.reshape(-1, 3)

    # lines computed frmo pts2
    lines2 = cv2.computeCorrespondEpilines(pts2_epi, 1, F)
    lines2 = lines2.reshape(-1, 3)

    # overlay lines1 on pts2
    fig = plt.figure('Epilines of set 1 on image 2')
    for r in lines1:
        a, b, c = r[0], r[1], r[2]
        x = np.linspace(-100, 100, 5)
        y = ((-c) - (a * x)) / b
        plt.plot(x, y)

    x = []
    y = []
    for p in pts2:
        x.append(p[0])
        y.append(p[1])

    plt.plot(x, y, 'r.')

    plt.show()

    # overlay lines2 on pts1
    fig = plt.figure('Epilines of set 2 on image 1')
    for r in lines2:
        a, b, c = r[0], r[1], r[2]
        x = np.linspace(-100, 100, 5)
        y = ((-c) - (a * x)) / b
        plt.plot(x, y)

    x = []
    y = []
    for p in pts1:
        x.append(p[0])
        y.append(p[1])

    plt.plot(x, y, 'r.')

    plt.show()


def is_singular(a):
    det = np.linalg.det(a)
    s = not is_invertible(a)
    print "> Singular:", s


def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def testEssentialReln(E, nh_pts1, nh_pts2):
    # check that x'Ex = 0 for normalised, homog coords x x'
    E = np.mat(E)
    is_singular(E)

    err = 0
    for i in range(0, len(nh_pts1)):
        err += abs(np.mat(nh_pts1[i]) * E * np.mat(nh_pts2[i]).T)

    err = err[0, 0] / len(nh_pts1)
    print "> avg error in x'Ex = 0:", err


def testRtCombo(R, t, pts1, pts2):
    P1 = BoringCameraArray()
    P2 = CameraArray(R, t)
    points3d = []

    for i in range(0, len(pts1)):
        x1 = pts1[i][0]
        y1 = pts1[i][0]
        x2 = pts2[i][0]
        y2 = pts2[i][0]

        u1 = Point(x1, y1)
        u2 = Point(x2, y2)

        X = tri.LinearTriangulation(P1, u1, P2, u2)
        points3d.append(X[1])

    for point in points3d:
        if point[2] < 0:
            return False

    return True


# used for checking the triangulation - provide UNNORMALISED DATA
def reprojectionError(K1, P1_mat, K2, P2_mat, pts1, pts2, points3d):

    new = np.zeros((len(points3d), 4))
    for i, point in enumerate(points3d):
        new[i][0] = point[0]
        new[i][1] = point[1]
        new[i][2] = point[2]
        new[i][3] = 1
    total = 0

    reprojected1 = []
    reprojected2 = []

    # for each 3d point
    for i, X in enumerate(new):
        # x_2d = K * P * X_3d
        xp1 = K1 * P1_mat * np.mat(X).T
        xp2 = K2 * P2_mat * np.mat(X).T

        # normalise the projected (homogenous) coordinates
        # (x,y,1) = (xz,yz,z) / z
        xp1 = xp1 / xp1[2]
        xp2 = xp2 / xp2[2]

        reprojected1.append(xp1)
        reprojected2.append(xp2)

        # and get the orginally measured points
        x1 = pts1[i]
        x2 = pts2[i]

        # difference between them is:
        dist1 = math.hypot(xp1[0] - x1[0], xp1[1] - x1[1])
        dist2 = math.hypot(xp2[0] - x2[0], xp2[1] - x2[1])

        total += dist1 + dist2

    print "\n> avg reprojection error:", total / (2 * len(points3d))

    plotReprojection(reprojected1, reprojected2, pts1, pts2)


def plotReprojection(proj1, proj2, meas1, meas2):

    proj1_x = [point[0] for point in proj1]
    proj1_y = [point[1] for point in proj1]

    meas1_x = [point[0] for point in meas1]
    meas1_y = [point[1] for point in meas1]

    fig = plt.figure('Reprojection of Reconstruction onto Image 1')
    plt.scatter(proj1_x, proj1_y, color='r')
    plt.scatter(meas1_x, meas1_y, color='b')
    plt.show()

    proj2_x = [point[0] for point in proj2]
    proj2_y = [point[1] for point in proj2]

    meas2_x = [point[0] for point in meas2]
    meas2_y = [point[1] for point in meas2]

    fig = plt.figure('Reprojection of Reconstruction onto Image 2')
    plt.scatter(proj2_x, proj2_y, color='r')
    plt.scatter(meas2_x, meas2_y, color='b')
    plt.show()


def BoringCameraArray():
    P = np.zeros((3, 4), dtype='float32')
    P[0][0] = 1
    P[1][1] = 1
    P[2][2] = 1
    return P


# P = [R|t]
def CameraArray(R, t):
    # just tack t on as a column to the end of R
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
    # print "\n> Triangulated data:"
    # for point in objectPoints:
    #     print point[0], point[1], point[2]

    all_x = [point[0] for point in objectPoints]
    all_y = [point[1] for point in objectPoints]
    all_z = [point[2] for point in objectPoints]

    fig = plt.figure('3D Reconstruction (Scale ambiguity)')
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(all_x, all_y, all_z, zdir='z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


run()

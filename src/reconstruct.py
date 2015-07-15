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
K1 = np.mat(struc.CalibArray(602, 640, 360))  # gopro
K2 = np.mat(struc.CalibArray(1005, 640, 360))  # d5000

# 1 / RHS / gopro
pts1_raw = [[537.0, -289.5],
            [584.0, -299.0],
            [631.5, -317.0],
            [709.5, -375.0],
            [746.0, -414.5],
            [772.0, -455.5],
            [796.5, -498.0],
            [833.5, -582.5],
            [844.5, -632.0],
            [859.5, -625.5],
            [874.0, -594.5],
            [899.5, -559.0],
            [911.5, -550.0],
            [929.0, -547.0],
            [945.0, -559.5],
            [949.5, -576.0],
            [957.0, -574.0],
            [966.0, -564.5],
            [980.5, -554.5],
            [986.0, -555.0],
            [992.0, -554.5],
            [997.0, -551.0],
            [1010.0, -543.5],
            [1015.0, -540.5],
            [1019.0, -538.5],
            [1023.5, -533.5],
            [1031.0, -530.5],
            [1034.5, -525.0],
            [1038.5, -521.0],
            [1041.5, -521.0],
            [1047.5, -520.0],
            [1050.5, -515.0],
            [1053.0, -515.0],
            [1055.0, -514.5],
            [1059.5, -510.5],
            [1063.0, -508.5],
            [1064.5, -507.5],
            [1066.5, -507.0],
            [1070.5, -504.0],
            [1072.0, -500.5],
            [1074.5, -500.5],
            [1076.0, -500.0],
            [1079.5, -498.0],
            [1083.5, -498.0],
            [1086.5, -498.0],
            [1090.0, -498.0],
            [1096.5, -500.0]]

# 2 / LHS / d5000
pts2_raw = [[811.5, -212.0],
            [763.5, -242.0],
            [715.5, -277.0],
            [675.5, -319.0],
            [642.5, -380.5],
            [610.0, -447.5],
            [581.5, -517.0],
            [551.5, -616.5],
            [558.5, -585.0],
            [561.0, -528.5],
            [543.0, -535.5],
            [528.5, -506.0],
            [516.0, -492.0],
            [499.0, -481.0],
            [486.0, -479.0],
            [474.5, -487.5],
            [464.0, -499.0],
            [454.0, -519.0],
            [450.5, -544.5],
            [444.5, -507.5],
            [436.0, -496.5],
            [429.0, -495.5],
            [425.5, -504.5],
            [415.5, -497.5],
            [408.5, -493.0],
            [403.0, -482.5],
            [395.5, -485.0],
            [390.0, -476.0],
            [386.0, -464.5],
            [378.5, -468.0],
            [373.5, -462.5],
            [369.0, -460.0],
            [368.0, -460.5],
            [362.5, -459.5],
            [359.5, -457.5],
            [357.5, -456.0],
            [357.5, -453.5],
            [351.0, -450.0],
            [348.5, -449.0],
            [347.0, -444.5],
            [344.5, -444.5],
            [341.0, -444.0],
            [337.5, -442.5],
            [336.5, -441.5],
            [337.0, -440.0],
            [336.0, -438.0],
            [343.0, -437.0]]

# Image coords: (x, y)
pts1 = np.array(pts1_raw, dtype='float32')
pts2 = np.array(pts2_raw, dtype='float32')

# Normalised homogenous image coords: (x, y, 1)
norm_pts1 = struc.normalise_homogenise(pts1, K1)
norm_pts2 = struc.normalise_homogenise(pts2, K2)

# Arrays FOR Rt computation
W, W_inv = struc.initWarrays()  # HZ 9.13


def run():
    # FUNDAMENTAL MATRIX
    F = getFundamentalMatrix(pts1, pts2)

    # ESSENTIAL MATRIX (HZ 9.12)
    E, w, u, vt = getEssentialMatrix(F, K1, K2)

    # CONSTRAINED ESSENTIAL MATRIX
    E_prime, w2, u2, vt2 = getConstrainedEssentialMatrix(u, vt)

    # PROJECTION/CAMERA MATRICES from E or E_prime (HZ 9.6.2)
    P1, P2 = getNormalisedPMatrices(E, u, vt)
    P1_mat = np.mat(P1)
    P2_mat = np.mat(P2)

    # FULL PROJECTION MATRICES (with K) P = K[Rt]
    KP1 = K1 * P1_mat
    KP2 = K2 * P2_mat

    # TRIANGULATION
    points3d = triangulateLS(KP1, KP2, pts1, pts2)
    # points4d = triangulateCV(KP1, KP2, pts1, pts2)

    # PLOTTING
    plot3D(points3d)
    reprojectionError(K1, P1_mat, K2, P2_mat, pts1, pts2, points3d)


def getFundamentalMatrix(pts1, pts2):
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

    return R, t


def triangulateLS(P1, P2, pts1, pts2):
    points3d = []

    for i in range(0, len(pts1)):
        x1 = pts1[i][0]
        y1 = pts1[i][0]
        x2 = pts2[i][0]
        y2 = pts2[i][0]

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
    pts1 = cv2.convertPointsToHomogeneous(pts1)
    pts2 = cv2.convertPointsToHomogeneous(pts2)
    err = 0
    total = 0
    for i in range(0, len(pts1)):
        # total += abs(pts1[i][0, 0]) + abs(pts1[i][0, 1]) + \
            # abs(pts2[i][0, 0]) + abs(pts2[i][0, 1])
        err += abs(np.mat(pts1[i]) * F * np.mat(pts2[i]).T)

    # avg = total / (4 * len(pts1))
    err = err[0, 0] / len(pts1)
    # per = (err / avg) * 100
    print "> avg error in xFx:", err
    # print "> typical x value is", avg
    # print "> typical %2.2f%% error" % per


def testEssentialReln(E, nh_pts1, nh_pts2):
    # check that x'Ex = 0 for normalised, homog coords x x'
    err = 0
    total = 0
    E = np.mat(E)

    for i in range(0, len(nh_pts1)):
        total += abs(nh_pts1[i][0]) + abs(nh_pts1[i][1]) + \
            abs(nh_pts2[i][0]) + abs(nh_pts2[i][1])
        err += np.mat(nh_pts1[i]) * E * np.mat(nh_pts2[i]).T

    avg = total / (4 * len(nh_pts1))
    err = err[0, 0] / len(nh_pts1)
    per = (err / avg) * 100
    print "> avg normalised px error in x'Ex:", err
    print "> typical x value is", avg
    print "> typical %2.2f%% error" % per


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

    # for each 3d point
    for i, X in enumerate(new):
        # x_2d = K * P * X_3d
        xp1 = K1 * P1_mat * np.mat(X).T
        xp2 = K2 * P2_mat * np.mat(X).T

        # normalise the projected (homogenous) coordinates
        # (x,y,1) = (xz,yz,z) / z
        xp1 = xp1 / xp1[2]
        xp2 = xp2 / xp2[2]

        # and get the orginally measured points
        x1 = pts1[i]
        x2 = pts2[i]

        # difference between them is:
        dist1 = math.hypot(xp1[0] - x1[0], xp1[1] - x1[1])
        dist2 = math.hypot(xp2[0] - x2[0], xp2[1] - x2[1])

        total += dist1 + dist2

    print "\n> avg reprojection error:", \
        total / (2 * len(points3d))


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
    print "\n> Triangulated data:"
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


run()

#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
import random

import fundamental as fund
import triangulation as tri
import structureTools as tools
import plotting as plot

random.seed()
np.set_printoptions(suppress=True)
plt.style.use('ggplot')

Point = namedtuple("Point", "x y")


# add some random noise to n image point set
def addNoise(a, b, points):
    new = []
    for p in points:
        n0 = p[0] + random.uniform(a, b)
        n1 = p[1] + random.uniform(a, b)
        n = [n0, n1]
        new.append(n)

    return np.array(new, dtype='float32')


# Get point correspondeces (1+2) from subdir
# Optionally: original 3d set, correspondences to be reconstructed (3+4)
def getData(folder):
    path = 'simulation_data/' + str(folder) + '/'
    pts1 = []
    pts2 = []
    pts3 = []
    pts4 = []
    original_3Ddata = []

    with open(path + 'pts1.txt') as datafile:
        data = datafile.read()
        datafile.close()

    data = data.split('\n')
    for row in data:
        x = float(row.split()[0])
        y = float(row.split()[1])
        pts1.append([x, y])

    with open(path + 'pts2.txt') as datafile:
        data = datafile.read()
        datafile.close()

    data = data.split('\n')
    for row in data:
        x = float(row.split()[0])
        y = float(row.split()[1])
        pts2.append([x, y])

    try:
        with open(path + '3d.txt') as datafile:
            data = datafile.read()
            datafile.close()

        data = data.split('\n')
        for row in data:
            x = float(row.split()[0])
            y = float(row.split()[1])
            z = float(row.split()[2])
            original_3Ddata.append([x, y, z])

    except IOError:
        print "> No 3D reference data provided. Not a simulation."

    try:
        with open(path + 'pts3.txt') as datafile:
            data = datafile.read()
            datafile.close()

        data = data.split('\n')
        for row in data:
            x = float(row.split()[0])
            y = float(row.split()[1])
            pts3.append([x, y])

        with open(path + 'pts4.txt') as datafile:
            data = datafile.read()
            datafile.close()

        data = data.split('\n')
        for row in data:
            x = float(row.split()[0])
            y = float(row.split()[1])
            pts4.append([x, y])

    except IOError:
        print "> No reconstruction points provided. Using full point set."
        pts3 = pts1
        pts4 = pts2

    return original_3Ddata, pts1, pts2, pts3, pts4

try:
    sim = sys.argv[1]
except IndexError:
    sim = 1

original_3Ddata, pts1_raw, pts2_raw, pts3_raw, pts4_raw = getData(sim)

# Image coords: (x, y)
pts1 = np.array(pts1_raw, dtype='float32')
pts2 = np.array(pts2_raw, dtype='float32')
pts3 = np.array(pts3_raw, dtype='float32')
pts4 = np.array(pts4_raw, dtype='float32')

print pts1[0]
# pts1 = addNoise(0, 0.5, pts1)
# pts2 = addNoise(0, 0.5, pts2)
print pts1[0]

# Calibration matrices:
K1 = np.mat(tools.CalibArray(5, 5, 5), dtype='float32')
K2 = np.mat(tools.CalibArray(5, 5, 5), dtype='float32')

# Normalised homogenous image coords: (x, y, 1)
norm_pts1 = tools.normalise_homogenise(pts1, K1)
norm_pts2 = tools.normalise_homogenise(pts2, K2)

# Inhomogenous but normalised K_inv(x, y) (for if you want to calc E directly)
inhomog_norm_pts1 = np.delete(norm_pts1, 2, 1)
inhomog_norm_pts2 = np.delete(norm_pts2, 2, 1)

# Arrays FOR Rt computation
W, W_inv, Z = tools.initWZarrays()  # HZ 9.13


def run():
    plot.plot3D(original_3Ddata, 'Original 3D Data')
    plot.plot2D(pts1_raw, 'First image')
    plot.plot2D(pts2_raw, 'Second image')

    # FUNDAMENTAL MATRIX
    F = getFundamentalMatrix(pts1, pts2)

    # print pts1, pts1.shape
    # new = pts1.reshape((1, -1))
    # print new

    # sys.exit()
    # newPoints1, newPoints2 = cv2.correctMatches(F, pts1, pts2)

    # ESSENTIAL MATRIX (HZ 9.12)
    E, w, u, vt = getEssentialMatrix(F, K1, K2)

    # PROJECTION/CAMERA MATRICES from E (HZ 9.6.2)
    P1, P2 = getNormalisedPMatrices(u, vt)
    P1_mat = np.mat(P1)
    P2_mat = np.mat(P2)

    # FULL PROJECTION MATRICES (with K) P = K[Rt]
    KP1 = K1 * P1_mat
    KP2 = K2 * P2_mat

    print "\n> KP1:\n", KP1
    print "\n> KP2:\n", KP2

    # TRIANGULATION
    p3d_ls = triangulateLS(KP1, KP2, pts3, pts4)

    # alternative triangulation
    p3d_cv = triangulateCV(KP1, KP2, pts3, pts4)

    # PLOTTING
    plot.plot3D(p3d_cv, '3D Reconstruction (Scale ambiguity)')
    reprojectionError(K1, P1_mat, K2, P2_mat, p3d_cv)


# get the Fundamental matrix by the normalised eight point algorithm
def getFundamentalMatrix(pts1, pts2):

    # 8point normalisation
    # pts1_, T1 = eightPointNormalisation(pts1)
    # pts2_, T2 = eightPointNormalisation(pts2)

    # plot.plot2D(pts1, pts1_, '8pt Normalisation on Image 1')
    # plot.plot2D(pts2, pts2_, '8pt Normalisation on Image 2')

    # normalised 8-point algorithm
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv.CV_FM_8POINT)
    tools.is_singular(F)

    # denormalise
    # F = T2.T * np.mat(F_) * T1
    # F = F / F[2, 2]

    # test on original coordinates
    print "\n> Fundamental:\n", F
    fund.testFundamentalReln(F, pts1, pts2)
    return F


def getEssentialMatrix(F, K1, K2):

    E = K1.T * np.mat(F) * K2
    print "\n> Essential:\n", E

    fund.testEssentialReln(E, norm_pts1, norm_pts2)
    s, u, vt = cv2.SVDecomp(E)

    print "> SVDecomp(E):"
    print "u:\n", u
    print "vt:\n", vt
    print "\n> Singular values:\n", s
    return E, s, u, vt


# https://en.wikipedia.org/wiki/Eight-point_algorithm#Step_3:_Enforcing_the_internal_constraint
def getConstrainedEssentialMatrix(u, vt):
    diag = np.mat(np.diag([1, 1, 0]))

    E_prime = np.mat(u) * diag * np.mat(vt)
    print "\n> Constrained Essential = u * diag(1,1,0) * vt:\n", E_prime
    fund.testEssentialReln(E_prime, norm_pts1, norm_pts2)

    s2, u2, vt2 = cv2.SVDecomp(E_prime)
    print "\n> Singular values:\n", s2

    return E_prime, s2, u2, vt2


def getNormalisedPMatrices(u, vt):
    R1 = np.mat(u) * np.mat(W) * np.mat(vt)
    R2 = np.mat(u) * np.mat(W.T) * np.mat(vt)

    # negate R if det(R) negative
    if np.linalg.det(R1) < 0:
        R1 = -1 * R1

    if np.linalg.det(R2) < 0:
        R2 = -1 * R2

    t1 = u[:, 2]
    t2 = -1 * u[:, 2]

    R, t = getValidRtCombo(R1, R2, t1, t2)

    # NORMALISED CAMERA MATRICES P = [Rt]
    P1 = BoringCameraArray()  # I|0
    P2 = CameraArray(R, t)    # R|t

    print "\n> P1:\n", P1
    print "\n> P2:\n", P2

    return P1, P2


def getValidRtCombo(R1, R2, t1, t2):
    # enforce positive depth combination of Rt using normalised coords
    if testRtCombo(R1, t1, norm_pts1, norm_pts2):
        print "\n> R|t: R1 t1"
        R = R1
        t = t1

    elif testRtCombo(R1, t2, norm_pts1, norm_pts2):
        print "\n> R|t: R1 t2"
        R = R1
        t = t2

    elif testRtCombo(R2, t1, norm_pts1, norm_pts2):
        print "\n> R|t: R2 t1"
        R = R2
        t = t1

    elif testRtCombo(R2, t2, norm_pts1, norm_pts2):
        print "\n> R|t: R2 t2"
        R = R2
        t = t2

    else:
        print "ERROR: No positive depth Rt combination"
        sys.exit()

    print "R:\n", R
    print "t:\n", t
    return R, t


# which combination of R|t gives us a P pair that works geometrically
# ie: gives us a positive depth measure in both
def testRtCombo(R, t, norm_pts1, norm_pts2):
    P1 = BoringCameraArray()
    P2 = CameraArray(R, t)
    points3d = []

    for i in range(0, len(norm_pts1)):
        x1 = norm_pts1[i][0]
        y1 = norm_pts1[i][1]

        x2 = norm_pts2[i][0]
        y2 = norm_pts2[i][1]

        u1 = Point(x1, y1)
        u2 = Point(x2, y2)

        X = tri.LinearTriangulation(P1, u1, P2, u2)
        points3d.append(X[1])

    # check if any z coord is negative
    for point in points3d:
        if point[2] < 0:
            return False

    return True


# linear least squares triangulation for one 3-space point X
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
def triangulateCV(KP1, KP2, pts_a, pts_b):
    points4d = cv2.triangulatePoints(KP1, KP2, pts_a.T, pts_b.T)
    points4d = points4d.T

    points3d = cv2.convertPointsFromHomogeneous(points4d)
    points3d = points3d.tolist()
    points3d = tools.fixExtraneousParentheses(points3d)

    return points3d


# used for checking the triangulation - provide UNNORMALISED DATA
def reprojectionError(K1, P1_mat, K2, P2_mat, points3d):

    # Nx4 array for filling with homogeneous points
    new = np.zeros((len(points3d), 4))

    for i, point in enumerate(points3d):
        new[i][0] = point[0]
        new[i][1] = point[1]
        new[i][2] = point[2]
        new[i][3] = 1

    errors1 = []
    errors2 = []
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
        x1 = pts3[i]
        x2 = pts4[i]

        # difference between them is:
        dist1 = math.hypot(xp1[0] - x1[0], xp1[1] - x1[1])
        dist2 = math.hypot(xp2[0] - x2[0], xp2[1] - x2[1])
        errors1.append(dist1)
        errors2.append(dist2)

    avg1 = sum(errors1) / len(errors1)
    avg2 = sum(errors2) / len(errors2)

    print "\n> average reprojection error in image 1:", avg1
    print "\n> average reprojection error in image 2:", avg2

    plot.plotOrderedBar(errors1, 'Reprojection Error Image 1', 'Index', 'px')
    plot.plotOrderedBar(errors2, 'Reprojection Error Image 2', 'Index', 'px')

    plot.plot2D(reprojected1, pts3,
                'Reprojection of Reconstruction onto Image 1')
    plot.plot2D(reprojected2, pts4,
                'Reprojection of Reconstruction onto Image 2')


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

print "---------------------------------------------"
run()

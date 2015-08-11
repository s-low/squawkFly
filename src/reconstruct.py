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

simulation = False


def synchroniseAtApex(pts_1, pts_2):
    print "APEX SYNCHRONISATION"

    syncd1 = []
    syncd2 = []
    shorter = []
    longer = []
    short_flag = 0

    if len(pts_1) < len(pts_2):
        shorter = pts_1
        longer = pts_2
        short_flag = 1
    else:
        shorter = pts_2
        longer = pts_1
        short_flag = 2

    diff = len(longer) - len(shorter)
    print "Difference in lengths:", diff
    # find the highest y value in each point set
    apex1 = max(float(p[1]) for p in shorter)
    apex2 = max(float(p[1]) for p in longer)

    apex1_i = [i for i, y in enumerate(shorter) if y[1] == apex1]
    apex2_i = [i for i, y in enumerate(longer) if y[1] == apex2]

    print "\nAPEXES"
    print "Short:", apex1, apex1_i, "of", len(shorter)
    print "Long:", apex2, apex2_i, "of", len(longer)

    shift = apex2_i[0] - apex1_i[0]

    # remove the front end dangle
    print "\nShift by:", shift

    if shift > 0:
        longer = longer[shift:]
        print "Longer front trimmed, new length:", len(longer)
    else:
        shorter = shorter[abs(shift):]
        print "Shorter front trimmed, new length:", len(shorter)

    remainder = diff - shift

    # remove the rear end dangle
    if remainder >= 0:
        print "\nTrim longer by remainder:", remainder
        index = len(longer) - remainder
        print "Trim to index:", index
        longer = longer[:index]

    if remainder < 0:
        index = len(shorter) - abs(remainder)
        print "\nShift > diff in lengths, trim the shorter end to:", index
        shorter = shorter[:index]

    print "New length of shorter:", len(shorter)
    print "New length of longer:", len(longer)

    # find the highest y value in each point set
    apex1 = max(float(p[1]) for p in shorter)
    apex2 = max(float(p[1]) for p in longer)

    apex1_i = [i for i, y in enumerate(shorter) if y[1] == apex1]
    apex2_i = [i for i, y in enumerate(longer) if y[1] == apex2]

    print "\nNew apex positions:"
    print apex1, apex1_i
    print apex2, apex2_i

    if short_flag == 1:
        syncd1 = shorter
        syncd2 = longer
    else:
        syncd1 = longer
        syncd2 = shorter

    plot.plot2D(syncd1, name='First Synced Trajectory')
    plot.plot2D(syncd2, name='Second Synced Trajectory')

    return syncd1, syncd2


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
    path = 'tests/' + str(folder) + '/'
    pts1 = []
    pts2 = []
    pts3 = []
    pts4 = []
    postPts1 = []
    postPts2 = []
    data3D = []

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
            data3D.append([x, y, z])

        print "> 3D reference data provided. Simulation."

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

        rec_data = True
        print "> Designated reconstruction correspondences provided."

    except IOError:
        print "> No reconstruction points provided. Using full point set."
        pts3 = pts1
        pts4 = pts2
        rec_data = False

    try:
        with open(path + 'postPts1.txt') as datafile:
            data = datafile.read()
            datafile.close()
        data = data.split('\n')
        for row in data:
            x = float(row.split()[0])
            y = float(row.split()[1])
            postPts1.append([x, y])
    except IOError:
        pass

    try:
        with open(path + 'postPts2.txt') as datafile:
            data = datafile.read()
            datafile.close()
        data = data.split('\n')
        for row in data:
            x = float(row.split()[0])
            y = float(row.split()[1])
            postPts2.append([x, y])
    except IOError:
        pass

    return data3D, pts1, pts2, pts3, pts4, postPts1, postPts2, rec_data


def undistortData(points, K, d):

    points = np.array(points, dtype='float32').reshape((-1, 1, 2))
    points = cv2.undistortPoints(
        src=points, cameraMatrix=K, distCoeffs=d, P=K).tolist()

    points_ = []
    for p in points:
        points_.append(p[0])

    return points_


# INITIALISE ANY GLOBALLY AVAILABLE DATA
try:
    d = sys.argv[1]
except IndexError:
    d = 1

if d.isdigit():
    simulation = True

# Calibration matrices:
K1 = np.mat(tools.CalibArray(950, 640, -360), dtype='float32')  # lumix
K2 = np.mat(tools.CalibArray(1091, 640, -360), dtype='float32')  # g3

dist_coeffs1 = np.array([-0.039, 0.18, 0, 0, 0])
dist_coeffs2 = np.array([0.006, 0.558, 0, 0, 0])

# If one of the simulation folders, set the calib matrices to sim values
if simulation:
    K1 = np.mat(tools.CalibArray(1000, 640, 360), dtype='float32')
    K2 = np.mat(tools.CalibArray(1000, 640, 360), dtype='float32')

# get the data from file
data3D, pts1_raw, pts2_raw, pts3_raw, pts4_raw, postPts1, postPts2, rec_data = getData(
    d)

# undistort it
# pts1_raw = undistortData(pts1_raw, K1, dist_coeffs1)
# pts2_raw = undistortData(pts2_raw, K2, dist_coeffs2)
# pts3_raw = undistortData(pts3_raw, K1, dist_coeffs1)
# pts4_raw = undistortData(pts4_raw, K2, dist_coeffs2)

pts1 = []
pts2 = []
pts3 = []
pts4 = []

# Image coords: (x, y)
pts1 = np.array(pts1_raw, dtype='float32')
pts2 = np.array(pts2_raw, dtype='float32')
pts3 = np.array(pts3_raw, dtype='float32')
pts4 = np.array(pts4_raw, dtype='float32')
postPts1 = np.array(postPts1, dtype='float32')
postPts2 = np.array(postPts2, dtype='float32')


# using the trajectories themselves to calculate geometry
if rec_data is False and simulation is False:
    pts1, pts2 = synchroniseAtApex(pts1, pts2)
    pts3, pts4 = synchroniseAtApex(pts3, pts4)

# Normalised homogenous image coords: (x, y, 1)
norm_pts1 = tools.normalise_homogenise(pts1, K1)
norm_pts2 = tools.normalise_homogenise(pts2, K2)

# Inhomogenous but normalised K_inv(x, y) (for if you want to calc E
# directly)
inhomog_norm_pts1 = np.delete(norm_pts1, 2, 1)
inhomog_norm_pts2 = np.delete(norm_pts2, 2, 1)

# Arrays FOR Rt computation
W, W_inv, Z = tools.initWZarrays()  # HZ 9.13


def run():
    global pts3
    global pts4

    if simulation:
        plot.plot3D(data3D, 'Original 3D Data')

    plot.plot2D(pts1_raw, name='First Static Correspondences')
    plot.plot2D(pts2_raw, name='Second Static Correspondences')

    # FUNDAMENTAL MATRIX
    F = getFundamentalMatrix(pts1, pts2)

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

    # SYNCHRONISATION + CORRECTION
    if rec_data:
        pts3, pts4 = synchroniseAtApex(pts3, pts4)
        # pts3, pts4 = synchroniseGeometric(pts3, pts4, F)

        pts3 = pts3.reshape((1, -1, 2))
        pts4 = pts4.reshape((1, -1, 2))
        newPoints3, newPoints4 = cv2.correctMatches(F, pts3, pts4)
        pts3 = newPoints3.reshape((-1, 2))
        pts4 = newPoints4.reshape((-1, 2))

    elif simulation:
        pts3 = pts1
        pts4 = pts2

    # add the post point data into the reconstruction for context
    if len(postPts1) == 4:
        pts3_gp = np.concatenate((pts3, postPts1), axis=0)
        pts4_gp = np.concatenate((pts4, postPts2), axis=0)

    # TRIANGULATION
    p3d_ls = triangulateLS(KP1, KP2, pts3_gp, pts4_gp)

    # alternative triangulation
    goal_posts = triangulateCV(KP1, KP2, postPts1, postPts2)

    # with goal posts included
    p3d_cv_gp = triangulateCV(KP1, KP2, pts3_gp, pts4_gp)

    # just the trajectory
    p3d_cv = triangulateCV(KP1, KP2, pts3, pts4)

    # SCALING AND PLOTTING
    scale = getScale(goal_posts)
    scaled_gp_only = [[a * scale for a in inner] for inner in goal_posts]
    scaled_gp = [[a * scale for a in inner] for inner in p3d_cv_gp]
    scaled = [[a * scale for a in inner] for inner in p3d_cv]

    plot.plot3D(scaled_gp_only, 'Goal Posts')
    plot.plot3D(scaled_gp, '3D Reconstruction')
    reprojectionError(K1, P1_mat, K2, P2_mat, pts3_gp, pts4_gp, p3d_cv_gp)

    getSpeed(scaled)


# give the scaled up set of trajectory points, work out the point to point
# speed and write it to a file
def getSpeed(worldPoints):
    outfile = open('tests/' + d + '/speed.txt', 'w')
    first = worldPoints.pop(0)
    prev = first
    speeds = []
    for p in worldPoints:
        dist = sep3D(p, prev)

        # dist is m travelled in ~15ms
        speed = 58 * dist
        mph = 2.23693629 * speed
        outfile.write(str(speed) + ' ' + str(mph) + '\n')

        speeds.append(mph)

        prev = p

    outfile.close()

    # calculate range
    last = prev
    shotRange = int(sep3D(first, last))

    avg = int(sum(speeds) / len(speeds))
    print "> Distance Covered:", str(shotRange) + 'm'
    print "> Average speed: ", str(avg) + 'mph'

    outfile = open('tests/' + d + '/tracer_stats.txt', 'w')
    outfile.write(str(avg) + '\n')
    outfile.write(str(shotRange))
    outfile.close()


# get the Fundamental matrix by the normalised eight point algorithm
def getFundamentalMatrix(pts_1, pts_2):

    # 8point normalisation
    # pts1_, T1 = eightPointNormalisation(pts1)
    # pts2_, T2 = eightPointNormalisation(pts2)

    # plot.plot2D(pts1, pts1_, '8pt Normalisation on Image 1')
    # plot.plot2D(pts2, pts2_, '8pt Normalisation on Image 2')

    # normalised 8-point algorithm
    F, mask = cv2.findFundamentalMat(pts_1, pts_2, cv.CV_FM_8POINT)
    tools.is_singular(F)

    # F, pts_1, pts_2 = autoGetF()

    # denormalise
    # F = T2.T * np.mat(F_) * T1
    # F = F / F[2, 2]

    # test on original coordinates
    print "\n> Fundamental:\n", F
    fund.testFundamentalReln(F, pts_1, pts_2)
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
    print "> RT Test:"
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
        print point[2]
        if point[2] < -0.1:
            return False

    return True


# linear least squares triangulation for one 3-space point X
def triangulateLS(P1, P2, pts_1, pts_2):
    points3d = []

    for i in range(0, len(pts_1)):

        x1 = pts_1[i][0]
        y1 = pts_1[i][1]

        x2 = pts_2[i][0]
        y2 = pts_2[i][1]

        p1 = Point(x1, y1)
        p2 = Point(x2, y2)

        X = tri.LinearTriangulation(P1, p1, P2, p2)

        points3d.append(X[1])

    return points3d


# expects normalised points
def triangulateCV(KP1, KP2, pts_1, pts_2):

    points4d = cv2.triangulatePoints(KP1, KP2, pts_1.T, pts_2.T)
    points4d = points4d.T
    points3d = convertFromHomogeneous(points4d)
    points3d = points3d.tolist()

    return points3d


# because openCV is stupid
def convertFromHomogeneous(points):
    new = []
    for p in points:
        s = 1 / p[3]
        n = (s * p[0], s * p[1], s * p[2])
        new.append(n)

    new = np.array(new, dtype='float32')
    return new


# given corners 1, 2, 3, 4, work out the 3d scale factor.
def getScale(goalPosts):
    p1 = goalPosts[0]  # bottom left
    p2 = goalPosts[1]  # top left
    p3 = goalPosts[2]  # top right
    p4 = goalPosts[3]  # bottom right

    distances = []

    leftBar = sep3D(p1, p2)
    crossbar = sep3D(p2, p3)
    rightBar = sep3D(p3, p4)
    baseline = sep3D(p1, p4)

    print "left uprights:", leftBar, rightBar
    print "crossbars:", crossbar, baseline

    distances = [leftBar, rightBar, baseline, crossbar]

    a = (crossbar + baseline) / 2
    b = (leftBar + rightBar) / 2

    scale_a = 7.32 / a
    scale_b = 2.44 / b

    print "crossbar and bar scales:", scale_a, scale_b
    scale = (scale_a + scale_b) / 2

    print "avg scale:", scale

    return scale


# distance between two 3d coordinates
def sep3D(a, b):
    xa = a[0]
    ya = a[1]
    za = a[2]

    xb = b[0]
    yb = b[1]
    zb = b[2]

    dist = math.sqrt(((xa - xb) ** 2) + ((ya - yb) ** 2) + ((za - zb) ** 2))

    return dist


# used for checking the triangulation - provide UNNORMALISED DATA
def reprojectionError(K1, P1_mat, K2, P2_mat, pts_3, pts_4, points3d):

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
        x1 = pts_3[i]
        x2 = pts_4[i]

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

    plot.plot2D(reprojected1, pts_3,
                'Reprojection of Reconstruction onto Image 1',
                lims=(1280, -720))
    plot.plot2D(reprojected2, pts_4,
                'Reprojection of Reconstruction onto Image 2',
                lims=(1280, -720))


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


# given a set of point correspondences x x', adjust the alignment such
# that x'Fx = 0 is smallest. obeys the geometry most closely.
def synchroniseGeometric(pts_1, pts_2, F):

    print "> GEOMETRIC SYNCHRONISATION:"

    syncd1 = []
    syncd2 = []
    shorter = []
    longer = []
    short_flag = 0

    if len(pts_1) < len(pts_2):
        shorter = pts_1
        longer = pts_2
        short_flag = 1
    else:
        shorter = pts_2
        longer = pts_1
        short_flag = 2

    diff = len(longer) - len(shorter)
    print "Longer:", len(longer)
    print "Shorter:", len(shorter)
    print "Diff:", diff

    shorter_hom = cv2.convertPointsToHomogeneous(shorter)
    longer_hom = cv2.convertPointsToHomogeneous(longer)

    averages = []

    for offset in xrange(0, diff + 1):
        err = 0
        avg = 0
        print ""
        for i in xrange(0, len(shorter)):
            a = shorter_hom[i]
            b = longer_hom[i + offset]
            this_err = abs(np.mat(a) * F * np.mat(b).T)
            err += this_err
            print this_err

        avg = err / len(shorter)
        # print "Offset, Err:", offset, avg
        avg_off = (avg, offset)
        averages.append(avg_off)

    m = min(float(a[0]) for a in averages)

    ret = [item for item in averages if item[0] == m]

    print "Minimum:", m
    print "Offset:", ret[0][1]

    # trim the beginning of the longer list
    offset = ret[0][1]
    longer = longer[offset:]

    # trim its end
    tail = len(longer) - len(shorter)
    if tail != 0:
        longer = longer[:-tail]

    if short_flag == 1:
        syncd1 = shorter
        syncd2 = longer
    else:
        syncd1 = longer
        syncd2 = shorter

    print "Synched Trajectory Length:", len(longer), len(shorter)

    plot.plot2D(syncd1, name='First Synced Trajectory')
    plot.plot2D(syncd2, name='Second Synced Trajectory')

    return syncd1, syncd2


# Copyright 2013, Alexander Mordvintsev & Abid K
def autoGetCorrespondences(img1, img2):
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    auto_pts1 = []
    auto_pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            auto_pts2.append(kp2[m.trainIdx].pt)
            auto_pts1.append(kp1[m.queryIdx].pt)

    return np.float32(auto_pts1), np.float32(auto_pts2)


def autoGetF():
    img1 = cv2.imread('../res/mvb/d5000.png', 0)
    img2 = cv2.imread('../res/mvb/g3.png', 0)

    auto_pts1, auto_pts2 = autoGetCorrespondences(img1, img2)

    F, mask = cv2.findFundamentalMat(auto_pts1, auto_pts2, cv2.FM_LMEDS)

    # We select only inlier points
    auto_pts1 = auto_pts1[mask.ravel() == 1]
    auto_pts2 = auto_pts2[mask.ravel() == 1]

    return F, auto_pts1, auto_pts2


print "---------------------------------------------"
run()

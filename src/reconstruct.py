#!/usr/local/bin/python

import cv2
import cv2.cv as cv
import numpy as np


def CalibMatrix(focalLength, cx, cy):
    cameraMatrix = np.zeros((3, 3), dtype='float32')
    cameraMatrix[0][0] = focalLength
    cameraMatrix[1][1] = focalLength
    cameraMatrix[2][2] = 1
    cameraMatrix[0][2] = cx
    cameraMatrix[1][2] = cy
    return cameraMatrix


def initSVDmatrices():
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


def BoringCameraMatrix():
    P = np.zeros((3, 4), dtype='float32')
    P[0][0] = 1
    P[1][1] = 1
    P[2][2] = 1
    return P


def CameraMatrix(R, t):
    P = np.zeros((3, 4), dtype='float32')
    P[0][0] = R[0][0]
    P[0][1] = R[0][1]
    P[0][2] = R[0][2]
    P[0][3] = t[0]

    P[1][0] = R[1][0]
    P[1][1] = R[1][1]
    P[1][2] = R[1][2]
    P[1][3] = t[1]

    P[2][0] = R[2][0]
    P[2][1] = R[2][1]
    P[2][2] = R[2][2]
    P[2][3] = t[2]

    return P


def LinearTriangulation(u1, P1, u2, P2):
    # Ax = 0
    A = np.zeros((3, 4), dtype='float32')
    A[0][0] = u1.x * P1(2, 0) - P1(0, 0)
    A[0][1] = u1.x * P1(2, 1) - P1(0, 1)
    A[0][2] = u1.x * P1(2, 2) - P1(0, 2)

    A[1][0] = u1.y * P1(2, 0) - P1(1, 0)
    A[1][1] = u1.y * P1(2, 1) - P1(1, 1)
    A[1][2] = u1.y * P1(2, 2) - P1(1, 2)

    A[2][0] = u2.x * P2(2, 0) - P2(0, 0)
    A[2][1] = u2.x * P2(2, 1) - P2(0, 1)
    A[2][2] = u2.x * P2(2, 2) - P2(0, 2)

    A[3][0] = u2.y * P2(2, 0) - P2(1, 0)
    A[3][1] = u2.y * P2(2, 1) - P2(1, 1)
    A[3][2] = u2.y * P2(2, 2) - P2(1, 2)

    print A
    # B = np.zeros((4,3))

# Mat_ LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
#                    Matx34d P,       //camera 1 matrix
#                    Point3d u1,      //homogenous image point in 2nd camera
#                    Matx34d P1       //camera 2 matrix
#                                    )
# {
#     //build matrix A for homogenous equation system Ax = 0
#     //assume X = (x,y,z,1), for Linear-LS method
#     //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
#     Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
#           u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
#           u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
#           u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
#               );
#     Mat_ B = (Mat_(4,1) <<    -(u.x*P(2,3)    -P(0,3)),
#                       -(u.y*P(2,3)  -P(1,3)),
#                       -(u1.x*P1(2,3)    -P1(0,3)),
#                       -(u1.y*P1(2,3)    -P1(1,3)));

#     Mat_ X;
#     solve(A,B,X,DECOMP_SVD);

#     return X;
# }

# 1 = RHS
pts1_raw = [[423, 191],  # t_l
            [840, 217],  # t_r
            [422, 352],  # b_l
            [838, 377],  # b_r
            [325, 437],  # front_l
            [744, 464],  # front_r
            [288, 344],  # wide_l
            [974, 388]]  # wide_r

# 2 = LHS
pts2_raw = [[423, 192],  # t_l
            [841, 166],  # t_r
            [422, 358],  # b_l
            [839, 330],  # b_r
            [518, 440],  # front_l
            [934, 417],  # front_r
            [287, 363],  # wide_l
            [973, 320]]  # wide_r

pts1 = np.array(pts1_raw, dtype='float32')
pts2 = np.array(pts2_raw, dtype='float32')

# given calibrated cameras
K1 = CalibMatrix(50, 0, 0)
K2 = CalibMatrix(50, 0, 0)

# Find FUNDAMENTAL MATRIX from point correspondences
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)
print "Fundamental:\n", F

# turn it into the ESSENTIAL MATRIX using calibration matrices K1/K2
E = K2.T * F * K1  # where 1 corresponds to the camera with P1 = [I|0]
print "Essential:\n", E

# Get the CAMERA MATRICES from E
# where one of them has P1 = [I|0] get the second camera matrix P2 = [R|t]
# this relies on Hartley and Zisserman sec 9.6

W, W_inv = initSVDmatrices()
w_, u, v = cv2.SVDecomp(E)
R = u * W * v.T  # HZ 9.19
t = u[:, 2]      # translation is third col of U
P1 = BoringCameraMatrix()
P2 = CameraMatrix(R, t)

print "P1:\n", P1
print "P2:\n", P2

# LinearTriangulation(P1)

# triangulate the points (don't use cv2.triangulatePoints)

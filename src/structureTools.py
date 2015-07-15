# General tools for 3d reconstruction

import numpy as np
import cv2


def CalibArray(focalLength, cx, cy):
    calibArray = np.zeros((3, 3), dtype='float32')
    calibArray[0][0] = focalLength
    calibArray[1][1] = focalLength
    calibArray[2][2] = 1
    calibArray[0][2] = cx
    calibArray[1][2] = cy
    return calibArray


# Convert a set of (x, y) to homogenous coords (x, y, 1)
# then normalise: K_inv(x, y, 1)
def normalise_homogenise(pts, K):
    pts = cv2.convertPointsToHomogeneous(pts)
    K_inv = np.linalg.inv(K)

    n_pts = np.zeros((len(pts), 3))

    for i, x in enumerate(pts):
        x_normed = K_inv * x.T
        n_pts[i][0] = xnormed[0]
        n_pts[i][1] = xnormed[1]
        n_pts[i][2] = xnormed[2]

    return n_pts


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

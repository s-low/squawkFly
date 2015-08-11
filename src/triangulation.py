# Triangulation module for 3D reconstruction from stereo views

import numpy as np
import cv2

np.set_printoptions(suppress=True)


# supply P1 and P2 as MAT
def LinearTriangulation(P1, p1, P2, p2):

    # points u are normalised (x, y)
    A = np.zeros((4, 3), dtype='float32')
    B = np.zeros((4, 1), dtype='float32')
    X = np.zeros((3, 1), dtype='float32')

    # top two rows for point 1
    A[0][0] = p1.x * P1[2, 0] - P1[0, 0]
    A[0][1] = p1.x * P1[2, 1] - P1[0, 1]
    A[0][2] = p1.x * P1[2, 2] - P1[0, 2]

    A[1][0] = p1.y * P1[2, 0] - P1[1, 0]
    A[1][1] = p1.y * P1[2, 1] - P1[1, 1]
    A[1][2] = p1.y * P1[2, 2] - P1[1, 2]

    # repeat for point 2
    A[2][0] = p2.x * P2[2, 0] - P2[0, 0]
    A[2][1] = p2.x * P2[2, 1] - P2[0, 1]
    A[2][2] = p2.x * P2[2, 2] - P2[0, 2]

    A[3][0] = p2.y * P2[2, 0] - P2[1, 0]
    A[3][1] = p2.y * P2[2, 1] - P2[1, 1]
    A[3][2] = p2.y * P2[2, 2] - P2[1, 2]

    # AX = B
    B[0][0] = -(p1.x * P1[2, 3] - P1[0, 3])
    B[1][0] = -(p1.y * P1[2, 3] - P1[1, 3])
    B[2][0] = -(p2.x * P2[2, 3] - P2[0, 3])
    B[3][0] = -(p2.y * P2[2, 3] - P2[1, 3])

    X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return X


def IterativeLinearTriangulation(P1, u1, P2, u2):
    EPSILON = 2

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
        A[0][0] = (u1.x * P1[2, 0] - P1[0, 0]) / wi1
        A[0][1] = (u1.x * P1[2, 1] - P1[0, 1]) / wi1
        A[0][2] = (u1.x * P1[2, 2] - P1[0, 2]) / wi1

        A[1][0] = (u1.y * P1[2, 0] - P1[1, 0]) / wi1
        A[1][1] = (u1.y * P1[2, 1] - P1[1, 1]) / wi1
        A[1][2] = (u1.y * P1[2, 2] - P1[1, 2]) / wi1

        A[2][0] = (u2.x * P2[2, 0] - P2[0, 0]) / wi2
        A[2][1] = (u2.x * P2[2, 1] - P2[0, 1]) / wi2
        A[2][2] = (u2.x * P2[2, 2] - P2[0, 2]) / wi2

        A[3][0] = (u2.y * P2[2, 0] - P2[1, 0]) / wi2
        A[3][1] = (u2.y * P2[2, 1] - P2[1, 1]) / wi2
        A[3][2] = (u2.y * P2[2, 2] - P2[1, 2]) / wi2

        B[0][0] = -(u1.x * P1[2, 3] - P1[0, 3])
        B[0][0] = -(u1.y * P1[2, 3] - P1[1, 3])
        B[0][0] = -(u2.x * P2[2, 3] - P2[0, 3])
        B[3][0] = -(u2.y * P2[2, 3] - P2[1, 3])

        X_ = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
        X[0] = X_[0]
        X[1] = X_[1]
        X[2] = X_[2]
        X[3] = 1

    return X

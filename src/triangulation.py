''' triangulation.py

    Implementation of the linear least squares triangulation from
    Hartley and Zisserman Multiple-View Geometry in CV p330...
'''
import numpy as np
import cv2

np.set_printoptions(suppress=True)


# supply P1 and P2 as MAT
def LinearTriangulation(P1, p1, P2, p2):

    # points u are normalised (x, y)
    A = np.zeros((4, 3), dtype='float32')
    B = np.zeros((4, 1), dtype='float32')
    X = np.zeros((3, 1), dtype='float32')

    # Top two rows of A for P1 and p1
    A[0][0] = p1.x * P1[2, 0] - P1[0, 0]
    A[0][1] = p1.x * P1[2, 1] - P1[0, 1]
    A[0][2] = p1.x * P1[2, 2] - P1[0, 2]

    A[1][0] = p1.y * P1[2, 0] - P1[1, 0]
    A[1][1] = p1.y * P1[2, 1] - P1[1, 1]
    A[1][2] = p1.y * P1[2, 2] - P1[1, 2]

    # Bottom two rows for P2 and p2 (repeat of above)
    A[2][0] = p2.x * P2[2, 0] - P2[0, 0]
    A[2][1] = p2.x * P2[2, 1] - P2[0, 1]
    A[2][2] = p2.x * P2[2, 2] - P2[0, 2]

    A[3][0] = p2.y * P2[2, 0] - P2[1, 0]
    A[3][1] = p2.y * P2[2, 1] - P2[1, 1]
    A[3][2] = p2.y * P2[2, 2] - P2[1, 2]

    # AX = B. B:
    B[0][0] = -(p1.x * P1[2, 3] - P1[0, 3])
    B[1][0] = -(p1.y * P1[2, 3] - P1[1, 3])
    B[2][0] = -(p2.x * P2[2, 3] - P2[0, 3])
    B[3][0] = -(p2.y * P2[2, 3] - P2[1, 3])

    # Solve AX = B for X by SVD
    X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return X

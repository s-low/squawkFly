import cv2
import cv2.cv as cv
import numpy as np
import math
import plotting as plot
import structureTools as tools

''' fundamental.py

Methods relating exclusively to the fundamental matrix and epipolar geometry.

Helpful to have all of these in one file:

- Test a fundamental matrix(F, pts1, pts2, view_flag)
- Test an essential matrix(E, pts1, pts2)
- Compute the distance to an epiline(line, point)
- Compute Hartley's normalisation
- Check Hartley's normalisation

'''


def testFundamentalReln(F, pts_1, pts_2, view):
    # check that xFx = 0 for homogenenous coords x x'
    F = np.mat(F)
    tools.is_singular(F)

    pts1_hom = cv2.convertPointsToHomogeneous(pts_1)
    pts2_hom = cv2.convertPointsToHomogeneous(pts_2)

    errors = []
    sum_err = 0

    # forwards: pt1 * F * pt2 = ?
    for i in range(0, len(pts1_hom)):
        this_err = abs(np.mat(pts1_hom[i]) * F * np.mat(pts2_hom[i]).T)
        sum_err += this_err[0, 0]
        errors.append(this_err[0, 0])

    # backwards 2 * F * 1 = ?
    for i in range(0, len(pts2_hom)):
        this_err = abs(np.mat(pts2_hom[i]) * F * np.mat(pts1_hom[i]).T)
        sum_err += this_err[0, 0]
        errors.append(this_err[0, 0])

    # NB: although defining eqn is K'.T * F * K, this just means
    # row x grid x col or (3x1)(3x3)(1x3). here our points are already rows
    # so we have to transpose the last to get our column

    err = sum_err / (2 * len(pts1_hom))
    print "> x'Fx = 0:", err

    # inspect the error distribution
    if view:
        plot.plotOrderedBar(errors,
                            name='x\'Fx = 0 Test Results ',
                            ylabel='Deflection from zero',
                            xlabel='Point Index')

    # test the epilines
    pts1_epi = pts_1.reshape((-1, 1, 2))
    pts2_epi = pts_2.reshape((-1, 1, 2))

    # lines computed from pts1
    # arg 2/3 is a flag to transpose F (2) or not (1)
    lines1 = cv2.computeCorrespondEpilines(pts1_epi, 1, F)
    lines1 = lines1.reshape(-1, 3)

    # lines computed from pts2
    # NB: passing 2 at pos1 results in a transpose of F in the calculation
    lines2 = cv2.computeCorrespondEpilines(pts2_epi, 2, F)
    lines2 = lines2.reshape(-1, 3)

    distances2 = []
    for l, p in zip(lines1, pts_2):
        distances2.append(distanceToEpiline(l, p))

    distances1 = []
    for l, p in zip(lines2, pts_1):
        distances1.append(distanceToEpiline(l, p))

    # Average p-line distances
    avg1 = sum(distances1) / len(distances1)
    avg2 = sum(distances2) / len(distances2)

    std1 = np.std(distances1)
    std2 = np.std(distances2)

    # Append the two lists of p-line measures
    distances = distances1 + distances2
    avg = np.mean(distances)
    std = np.std(distances)

    print "> Average distance to epiline in image 1 and 2 (px):", avg1, avg2
    print "> Overall Average:", avg
    print "> Std Dev:", std

    if view:
        # Inspect the distributions
        plot.plotOrderedBar(distances1,
                            'Image 1: Point-Epiline Distances', 'Index', 'px')

        plot.plotOrderedBar(distances2,
                            'Image 2: Point-Epiline Distances', 'Index', 'px')

        # overlay lines2 on pts1
        plot.plotEpilines(lines2, pts_1, 1)

        # overlay lines1 on pts2
        plot.plotEpilines(lines1, pts_2, 2)

    return avg, std


# find the distance between an epiline and image point
def distanceToEpiline(line, pt):

    # ax + by + c = 0
    a, b, c = line[0], line[1], line[2]

    # image point coords
    x = pt[0]
    y = pt[1]

    # y = mx + k (epiline)
    m1 = -a / b
    k1 = -c / b

    # y = -1/m x + k2 (perpedicular line that runs through p(x,y))
    if m1 < 0.0001:
        # if line1 is horizontal in x then the other line is vertical in x
        m2 = 99999
    else:
        m2 = -1 / m1

    k2 = y - (m2 * x)

    # x at point of intersection:
    x_inter = (k2 - k1) / (m1 - m2)

    # y1(x_intercept) and y2(x_intercept) should be the same
    y_inter1 = (m1 * x_inter) + k1
    y_inter2 = (m2 * x_inter) + k2

    message = "Epiline and perp have different y at x_intercept:" + \
        str(y_inter1) + ' ' + str(y_inter2)

    assert(abs(y_inter1 - y_inter2) < 5), message

    # distance between p(x, y) and intersect(x, y)
    d = math.hypot(x - x_inter, y - y_inter1)

    return d


# check that x'Ex = 0 for normalised, homog coords x x'
def testEssentialReln(E, nh_pts1, nh_pts2):
    E = np.mat(E)
    tools.is_singular(E)

    err = 0
    for i in range(0, len(nh_pts1)):
        err += abs(np.mat(nh_pts1[i]) * E * np.mat(nh_pts2[i]).T)

    err = err[0, 0] / len(nh_pts1)
    print "> x'Ex = 0:", err


# translate and scale image points, return both points and the transformation T
def eightPointNormalisation(pts):
    print "> 8POINT NORMALISATION"

    cx = 0
    cy = 0
    pts_ = []

    for p in pts:
        cx += p[0]
        cy += p[1]

    cx = cx / len(pts)
    cy = cy / len(pts)

    # translation to (cx,cy) = (0,0)
    T = np.mat([[1, 0, -cx],
                [0, 1, -cy],
                [0, 0, 1]])

    print "Translate by:", -cx, -cy

    # now scale to rms_d = sqrt(2)
    total_d = 0
    for p in pts:
        d = math.hypot(p[0] - cx, p[1] - cy)
        total_d += (d * d)

    # square root of the mean of the squares
    rms_d = math.sqrt((total_d / len(pts)))

    scale_factor = math.sqrt(2) / rms_d
    print "Scale by:", scale_factor

    T = scale_factor * T
    T[2, 2] = 1
    print "T:\n", T

    # apply the transformation
    hom = cv2.convertPointsToHomogeneous(pts)
    for h in hom:
        h_ = T * h.T
        pts_.append(h_)

    pts_ = cv2.convertPointsFromHomogeneous(np.array(pts_, dtype='float32'))
    check8PointNormalisation(pts_)

    # make sure the normalised points are in the same format as original
    pts_r = []
    for p in pts_:
        pts_r.append(p[0])
    pts_r = np.array(pts_r, dtype='float32')

    return pts_r, T


# average distance from origin should be sqrt(2) and centroid = origin
def check8PointNormalisation(pts_):
    d_tot = 0
    cx = 0
    cy = 0
    for p in pts_:
        cx += p[0][0]
        cx += p[0][1]
        d = math.hypot(p[0][0], p[0][1])
        d_tot += d

    avg = d_tot / len(pts_)
    cx = cx / len(pts_)
    cy = cy / len(pts_)

    assert(avg - math.sqrt(2) < 0.01), "Scale factor is wrong"
    assert(cx < 0.1 and cy < 0.1), "Centroid not at origin"

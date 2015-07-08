#!/usr/local/bin/python

import cv2
import cv2.cv as cv
import numpy as np

pts1 = [[423, 191],  # t_l
        [840, 217],  # t_r
        [422, 352],  # b_l
        [838, 377],  # b_r
        [325, 437],  # front_l
        [744, 464],  # front_r
        [288, 344],  # wide_l
        [974, 388]]  # wide_r

pts2 = [[423, 192],  # t_l
        [841, 166],  # t_r
        [422, 358],  # b_l
        [839, 330],  # b_r
        [518, 440],  # front_l
        [934, 417],  # front_r
        [287, 363],  # wide_l
        [973, 320]]  # wide_r

pts1 = np.array(pts1, dtype='f8')
pts2 = np.array(pts2, dtype='f8')

f, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)

# reshape the array into twice it's length but only 1 wide
pts1_r = pts1.reshape((pts1.shape[0] * 2, 1))
pts2_r = pts2.reshape((pts2.shape[0] * 2, 1))

ret, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_r, pts2_r, f, (1280, 720))

pts1 = np.array(pts1, dtype='f8')

# print pts1
# print H1

# ---------------------------

# size = 3, 3, 2
# dst = np.zeros(size, dtype='f8')
# print dst.dtype

# dst = cv.fromarray(dst)
# m = cv.fromarray(H1)
# src = cv.fromarray(pts1)

# print "src.type:", src.type
# print "dst.type:", dst.type
# print "m.rows:", m.rows
# print "dst.channels:", dst.channels

# dest = cv.createMat(r, c, cv.CV_32FC1)
# src = cv.fromarray(your_np_array)
# cv.Convert(src, dest)

src = cv.CreateMat(8, 1, cv.CV_32FC2)
dst = cv.CreateMat(8, 1, cv.CV_32FC2)
m = cv.CreateMat(3, 3, cv.CV_32F)

cv.PerspectiveTransform(src, dst, m)

print np.asarray(src)
print np.asarray(dst)

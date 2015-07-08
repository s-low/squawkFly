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

pts1 = np.array(pts1, dtype='f4')
pts2 = np.array(pts2, dtype='f4')

f, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)

# reshape the array into twice it's length but only 1 wide
pts1_r = pts1.reshape((pts1.shape[0] * 2, 1))
pts2_r = pts2.reshape((pts2.shape[0] * 2, 1))

ret, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_r, pts2_r, f, (1280, 720))

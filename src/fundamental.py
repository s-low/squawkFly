#!/usr/local/bin/python

import cv2
import cv2.cv as cv
import numpy as np

lhs = cv2.imread('res/LHS.png', 0)
rhs = cv2.imread('res/RHS.png', 0)

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

f, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC)

lines1 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, f)
lines1 = lines1.reshape(-1, 3)

lines2 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 1, f)
lines2 = lines2.reshape(-1, 3)

# reshape the array into twice it's length but only 1 wide
pts1_r = pts1.reshape((pts1.shape[0] * 2, 1))
pts2_r = pts2.reshape((pts2.shape[0] * 2, 1))

ret, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_r, pts2_r, f, (720, 1280))

src1 = np.array(pts1, dtype='float32')
src2 = np.array(pts2, dtype='float32')
H1 = np.array(H1, dtype='float32')
H2 = np.array(H2, dtype='float32')
src1 = np.array([src1])  # this is HUGE thing - adds an extra channel
src2 = np.array([src2])

# transform the points such that their epilines are parallel with x axis
dst1 = cv2.perspectiveTransform(src1, H1)
dst2 = cv2.perspectiveTransform(src2, H2)

# get the new epilines for the transformed points:
f, mask = cv2.findFundamentalMat(dst1, dst2, cv2.RANSAC)

rect_lines1 = cv2.computeCorrespondEpilines(dst1.reshape(-1, 1, 2), 2, f)
rect_lines1 = rect_lines1.reshape(-1, 3)

rect_lines2 = cv2.computeCorrespondEpilines(dst2.reshape(-1, 1, 2), 1, f)
rect_lines2 = rect_lines2.reshape(-1, 3)

# two blank images
img1 = np.full((720, 1280, 3), 255, np.uint8)
img2 = np.full((720, 1280, 3), 255, np.uint8)

for r in lines1:
    x0, y0 = map(int, [0, -r[2] / r[1]])
    x1, y1 = map(int, [1280, -(r[2] + r[0] * 1280) / r[1]])
    cv2.line(lhs, (x0, y0), (x1, y1), (0, 0, 0), 1)

for r in lines2:
    x0, y0 = map(int, [0, -r[2] / r[1]])
    x1, y1 = map(int, [1280, -(r[2] + r[0] * 1280) / r[1]])
    cv2.line(rhs, (x0, y0), (x1, y1), (0, 0, 0), 1)

for r in lines2:
    x0, y0 = map(int, [0, -r[2] / r[1]])
    x1, y1 = map(int, [1280, -(r[2] + r[0] * 1280) / r[1]])
    cv2.line(img1, (x0, y0), (x1, y1), (255, 0, 0), 1)

for r in rect_lines2:
    x0, y0 = map(int, [0, -r[2] / r[1]])
    x1, y1 = map(int, [1280, -(r[2] + r[0] * 1280) / r[1]])
    cv2.line(img2, (x0, y0), (x1, y1), (255, 0, 0), 1)

l_tl = (pts1_raw[0][0], pts1_raw[0][1])
l_tr = (pts1_raw[1][0], pts1_raw[1][1])
l_bl = (pts1_raw[2][0], pts1_raw[2][1])
l_br = (pts1_raw[3][0], pts1_raw[3][1])
l_fl = (pts1_raw[4][0], pts1_raw[4][1])
l_fr = (pts1_raw[5][0], pts1_raw[5][1])
l_wl = (pts1_raw[6][0], pts1_raw[6][1])
l_wr = (pts1_raw[7][0], pts1_raw[7][1])

r_tl = (pts2_raw[0][0], pts2_raw[0][1])
r_tr = (pts2_raw[1][0], pts2_raw[1][1])
r_bl = (pts2_raw[2][0], pts2_raw[2][1])
r_br = (pts2_raw[3][0], pts2_raw[3][1])
r_fl = (pts2_raw[4][0], pts2_raw[4][1])
r_fr = (pts2_raw[5][0], pts2_raw[5][1])
r_wl = (pts2_raw[6][0], pts2_raw[6][1])
r_wr = (pts2_raw[7][0], pts2_raw[7][1])

cv2.line(img1, pt1=l_tl, pt2=l_tr, color=(0, 0, 0))
cv2.line(img1, pt1=l_tr, pt2=l_br, color=(0, 0, 0))
cv2.line(img1, pt1=l_tl, pt2=l_bl, color=(0, 0, 0))
cv2.line(img1, pt1=l_bl, pt2=l_br, color=(0, 0, 0))
cv2.line(img1, pt1=l_bl, pt2=l_fl, color=(0, 0, 0))
cv2.line(img1, pt1=l_br, pt2=l_fr, color=(0, 0, 0))
cv2.line(img1, pt1=l_fr, pt2=l_fl, color=(0, 0, 0))
cv2.line(img1, pt1=l_wr, pt2=l_br, color=(0, 0, 0))
cv2.line(img1, pt1=l_wl, pt2=l_bl, color=(0, 0, 0))

cv2.line(img1, pt1=r_tl, pt2=r_tr, color=(0, 0, 255))
cv2.line(img1, pt1=r_tr, pt2=r_br, color=(0, 0, 255))
cv2.line(img1, pt1=r_tl, pt2=r_bl, color=(0, 0, 255))
cv2.line(img1, pt1=r_bl, pt2=r_br, color=(0, 0, 255))
cv2.line(img1, pt1=r_bl, pt2=r_fl, color=(0, 0, 255))
cv2.line(img1, pt1=r_br, pt2=r_fr, color=(0, 0, 255))
cv2.line(img1, pt1=r_fr, pt2=r_fl, color=(0, 0, 255))
cv2.line(img1, pt1=r_wr, pt2=r_br, color=(0, 0, 255))
cv2.line(img1, pt1=r_wl, pt2=r_bl, color=(0, 0, 255))

for row in pts1_raw:
    x_curr = row[0]
    y_curr = row[1]
    cv2.circle(img1, (x_curr, y_curr), radius=4, color=(0, 0, 0), thickness=-1)

for row in pts2_raw:
    x = row[0]
    y = row[1]
    cv2.circle(img1, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

for row in dst1[0]:
    x = row[0]
    y = row[1]
    cv2.circle(img2, center=(x, y), radius=4, color=(0, 0, 0), thickness=-1)

for row in dst2[0]:
    x = row[0]
    y = row[1]
    cv2.circle(img2, center=(x, y), radius=4, color=(0, 0, 255), thickness=-1)

l_tl = (dst1[0][0][0], dst1[0][0][1])
l_tr = (dst1[0][1][0], dst1[0][1][1])
l_bl = (dst1[0][2][0], dst1[0][2][1])
l_br = (dst1[0][3][0], dst1[0][3][1])
l_fl = (dst1[0][4][0], dst1[0][4][1])
l_fr = (dst1[0][5][0], dst1[0][5][1])
l_wl = (dst1[0][6][0], dst1[0][6][1])
l_wr = (dst1[0][7][0], dst1[0][7][1])

r_tl = (dst2[0][0][0], dst2[0][0][1])
r_tr = (dst2[0][1][0], dst2[0][1][1])
r_bl = (dst2[0][2][0], dst2[0][2][1])
r_br = (dst2[0][3][0], dst2[0][3][1])
r_fl = (dst2[0][4][0], dst2[0][4][1])
r_fr = (dst2[0][5][0], dst2[0][5][1])
r_wl = (dst2[0][6][0], dst2[0][6][1])
r_wr = (dst2[0][7][0], dst2[0][7][1])

cv2.line(img2, pt1=l_tl, pt2=l_tr, color=(0, 0, 0))
cv2.line(img2, pt1=l_tr, pt2=l_br, color=(0, 0, 0))
cv2.line(img2, pt1=l_tl, pt2=l_bl, color=(0, 0, 0))
cv2.line(img2, pt1=l_bl, pt2=l_br, color=(0, 0, 0))
cv2.line(img2, pt1=l_bl, pt2=l_fl, color=(0, 0, 0))
cv2.line(img2, pt1=l_br, pt2=l_fr, color=(0, 0, 0))
cv2.line(img2, pt1=l_fr, pt2=l_fl, color=(0, 0, 0))
cv2.line(img2, pt1=l_wr, pt2=l_br, color=(0, 0, 0))
cv2.line(img2, pt1=l_wl, pt2=l_bl, color=(0, 0, 0))

cv2.line(img2, pt1=r_tl, pt2=r_tr, color=(0, 0, 255))
cv2.line(img2, pt1=r_tr, pt2=r_br, color=(0, 0, 255))
cv2.line(img2, pt1=r_tl, pt2=r_bl, color=(0, 0, 255))
cv2.line(img2, pt1=r_bl, pt2=r_br, color=(0, 0, 255))
cv2.line(img2, pt1=r_bl, pt2=r_fl, color=(0, 0, 255))
cv2.line(img2, pt1=r_br, pt2=r_fr, color=(0, 0, 255))
cv2.line(img2, pt1=r_fr, pt2=r_fl, color=(0, 0, 255))
cv2.line(img2, pt1=r_wr, pt2=r_br, color=(0, 0, 255))
cv2.line(img2, pt1=r_wl, pt2=r_bl, color=(0, 0, 255))

show = True
while show:
    cv2.imshow('LHS', lhs)
    cv2.imshow('RHS', rhs)
    cv2.imshow('Original with LHS epilines', img1)
    cv2.imshow('Rectified with LHS epilines', img2)
    cv2.waitKey()
    show = False

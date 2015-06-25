#!/usr/local/bin/python


import sys
import cv2
import cv2.cv as cv
import numpy as np
from kfilter import KFilter

kf = KFilter()
kf.setPostState(100,100,8,8)

print "\n-------------SET------------"
print "pre\n",np.asarray(kf.getPreState())
print "post\n",np.asarray(kf.getPostState())
print "transition\n",np.asarray(kf.kf.transition_matrix)
print "measurement\n",np.asarray(kf.kf.measurement_matrix)

kf.update(110,110)

print "\n--------UPDATED (110,110)------------"
# print "pre\n",np.asarray(kf.getPreState())
# print "post\n",np.asarray(kf.getPostState())
predicted = kf.getPredicted()
corrected = kf.getCorrected()

print "PREDICTED"
print " x:",predicted[0]
print " y:",predicted[1]
print "vx:",predicted[2]
print "vy:",predicted[3]

print "CORRECTED"
print " x:",corrected[0]
print " y:",corrected[1]
print "vx:",corrected[2]
print "vy:",corrected[3]

kf.update(120,120)

print "\n---------UPDATED (120,120)----------"
# print "pre\n",np.asarray(kf.getPreState())
# print "post\n",np.asarray(kf.getPostState())
predicted = kf.getPredicted()
corrected = kf.getCorrected()

print "PREDICTED"
print " x:",predicted[0]
print " y:",predicted[1]
print "vx:",predicted[2]
print "vy:",predicted[3]

print "CORRECTED"
print " x:",corrected[0]
print " y:",corrected[1]
print "vx:",corrected[2]
print "vy:",corrected[3]

kf.update(130,130)

print "\n--------UPDATED (130,130)---------"
# print "pre\n",np.asarray(kf.getPreState())
# print "post\n",np.asarray(kf.getPostState())
predicted = kf.getPredicted()
corrected = kf.getCorrected()

print "PREDICTED"
print " x:",predicted[0]
print " y:",predicted[1]
print "vx:",predicted[2]
print "vy:",predicted[3]

print "CORRECTED"
print " x:",corrected[0]
print " y:",corrected[1]
print "vx:",corrected[2]
print "vy:",corrected[3]


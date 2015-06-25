#!/usr/local/bin/python


import sys
import cv2
import cv2.cv as cv
import numpy as np
from kfilter import KFilter

kf = KFilter()

print "---------INITIALISED---------"
print "pre\n",np.asarray(kf.getPreState())
print "post\n",np.asarray(kf.getPostState())
print "transition\n",np.asarray(kf.kf.transition_matrix)
print "measurement\n",np.asarray(kf.kf.measurement_matrix)

# kf.setState(100,100,10,10)
kf.setPostState(100,100,10,10)

print "\n---------PRE STATE SET--------"
print "pre\n",np.asarray(kf.getPreState())
print "post\n",np.asarray(kf.getPostState())
print "transition\n",np.asarray(kf.kf.transition_matrix)
print "measurement\n",np.asarray(kf.kf.measurement_matrix)

kf.update(110,110)

print "\n-----------UPDATED------------"
print "pre\n",np.asarray(kf.getPreState())
print "post\n",np.asarray(kf.getPostState())
print "transition\n",np.asarray(kf.kf.transition_matrix)
print "measurement\n",np.asarray(kf.kf.measurement_matrix)
print "predicted\n",kf.getPredicted()
print "corrected\n",kf.getCorrected()


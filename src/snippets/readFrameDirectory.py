#!/usr/local/bin/python

''' readFrameDirectory.py

    tiny test script for investigating reading an image sequence.
'''

import cv2
import cv2.cv as cv
import sys
import os.path

cap = cv2.VideoCapture('../res/mvb/mvb1_d5000/%05d.png')

while (1):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        cv2.waitKey()
    else:
        break

cap.release()

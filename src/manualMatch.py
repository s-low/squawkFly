#!/usr/local/bin/python

'''
./manualMatch.py <infile> <outfile>

User interface for marking up correspondences manually. Input file is a pair of
images. One image is shown, and when a point is marked with a click the second
image is shown for comparison and click.
'''

import cv2
import cv2.cv as cv
import sys
import os


# mouse callback function
def click(event, x, y, flags, param):
    global counter
    global current
    if event == cv2.EVENT_LBUTTONDOWN:
        counter += 1

        if counter == 6:
            sys.exit()

        current = frames[counter]
        if counter > 1:
            cv2.circle(current, (x, y), 5, (255, 0, 0), -1)
            # make sure to invert y-coord
            string = str(x) + ' -' + str(y)
            outfile.write(string + '\n')

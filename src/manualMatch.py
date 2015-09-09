#!/usr/local/bin/python

''' manualMatch.py

User interface for marking up stereo correspondences manually. Input is
a pair of images. One image is shown, and when a point is marked with a
click the second image is shown for comparison and click.

User actually supplies either a video or image sequence, and the first
frame is sampled for matching.

correspondences are written to two different files in an ordered list.

arg1 = image sequence or video 1
arg2 = image sequence or video 2
arg3 = outfile1
arg4 = outfile2

'''

import cv2
import cv2.cv as cv
import sys
import os


# mouse callback function, record the click coords and swap image
def click(event, x, y, flags, param):
    global counter
    global current
    if event == cv2.EVENT_LBUTTONDOWN:

        # alternate between first and second image on click
        counter += 1
        counter = counter % len(frames)
        current = frames[counter]

        # make sure to invert y-coord
        string = str(x) + ' -' + str(y)
        if counter == 1:
            outfile1.write(string + '\n')
        elif counter == 0:
            outfile2.write(string + '\n')


# Just colors and fonts we want to use
blue = (255, 0, 0)
red = (0, 0, 255)
white = (255, 255, 255)
font = cv2.FONT_HERSHEY_DUPLEX
counter = 0

# Files:
if len(sys.argv) != 5:
    print "Usage : /manualMatch.py <images_1> <images_2> <outfile1> <outfile2>"
    sys.exit(0)

image1 = sys.argv[1]
image2 = sys.argv[2]
out1 = sys.argv[3]
out2 = sys.argv[4]

outfile1 = open(out1, 'w')
outfile2 = open(out2, 'w')

# Supplied path can be a directory containing an image sequence: 00001.png...
if os.path.isdir(image1):
    image1 = image1 + '/frame_%05d.png'
if os.path.isdir(image2):
    image2 = image2 + '/frame_%05d.png'

# Get the first frame of each video
cap = cv2.VideoCapture(image1)
ret, img1 = cap.read()
cap.release()

cap = cv2.VideoCapture(image2)
ret, img2 = cap.read()
cap.release()

# label the images
cv2.putText(img1, "Image 1", (30, 45),
            fontFace=font,
            fontScale=1.5,
            color=red,
            thickness=2)

cv2.putText(img2, "Image 2", (30, 45),
            fontFace=font,
            fontScale=1.5,
            color=red,
            thickness=2)

frames = (img1, img2)
current = img1

cv2.namedWindow('Click on the same point in both images')
cv2.setMouseCallback('Click on the same point in both images', click)

while(1):
    cv2.imshow('Click on the same point in both images', current)
    if cv2.waitKey(20) & 0xFF == 113:
        break

cap.release()
cv2.destroyAllWindows()
outfile1.close()
outfile2.close()

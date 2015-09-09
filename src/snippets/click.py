#!/usr/local/bin/python

''' click.py
Simple python and opencv script to detect and log the locations of mouse
clicks on an image. writes the coordinates to file.

arg1 = image file name
outfile is 'data/temp_clicks.txt'
'''

import cv2
import cv2.cv as cv
import sys


# mouse callback function
def click(event, x, y, flags, param):
    global counter

    red = (0, 0, 255)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_PLAIN

    if event == cv2.EVENT_LBUTTONDOWN:
        counter += 1
        string = str(x) + ' -' + str(y)

        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(img, str(counter), (x, y), font, 2, red)
        outfile.write(string + '\n')

# Supply the image as arg1
filename = sys.argv[1]
outfile = open('data/temp_clicks.txt', 'w')
counter = 0

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', click)

img = cv2.imread(filename)

while(1):
    cv2.imshow('Image', img)
    if cv2.waitKey(20) & 0xFF == 113:
        break

cv2.destroyAllWindows()
outfile.close()

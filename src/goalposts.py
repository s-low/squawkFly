#!/usr/local/bin/python

# MAIN DETECTION TEST FILE
import sys
import cv2
import numpy as np
import os.path


def show(image):
    while(1):
        cv2.imshow('Image', image)
        if cv2.waitKey(20) & 0xFF == 113:
            break

filename = sys.argv[1]

image = cv2.imread(filename)

show(image)

lower = [0, 0, 90]
upper = [40, 40, 255]

lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

show(output)

grayed = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
show(grayed)

ret, thresh = cv2.threshold(grayed, 10, 255, cv2.THRESH_BINARY)
show(thresh)

kernel = np.ones((5, 5), np.uint8)
thresh = cv2.dilate(thresh, kernel)

show(thresh)

kernel = np.ones((11, 11), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

show(thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
show(image)

max_area = 500
min_area = 5

for contour in contours:
    area = cv2.retval = cv2.contourArea(contour)

    if area < max_area and area > min_area:
        x, y, w, h = cv2.boundingRect(contour)
        print x, y
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cx = x + float(w) / 2.0
        cy = -1 * (y + float(h) / 2.0)

show(image)

cv2.destroyAllWindows()

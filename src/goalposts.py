#!/usr/local/bin/python

# MAIN DETECTION TEST FILE
import sys
import cv2
import numpy as np
import os.path

filename = "../res/goalposts_red.png"

image = cv2.imread(filename)


while(1):
    cv2.imshow('Image', image)
    if cv2.waitKey(20) & 0xFF == 113:
        break

lower = [0, 0, 200]
upper = [50, 50, 255]

lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

while(1):
    cv2.imshow('Image', output)
    if cv2.waitKey(20) & 0xFF == 113:
        break

cv2.destroyAllWindows()

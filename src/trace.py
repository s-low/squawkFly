#!/usr/local/bin/python

import sys
import cv2
import numpy as np
import os.path

# two command line arguments:
# 1. A video clip or image sequence
# 2. The the 2D trajectory corresponding to that clip

if len(sys.argv) != 3:
    print "Usage : python trace.py <image_sequence> <trajectory>"
    sys.exit(0)

clip = sys.argv[1]
trajectory = sys.argv[2]

if os.path.isdir(clip):
    print "> Input Type: Image Sequence"
    clip = clip + '/frame_%05d.png'
else:
    print "> Input Type: Video File"

# Get trajectory data
with open(trajectory) as datafile:
    data = datafile.read()
    datafile.close()

all_x = []
all_y = []
all_f = []

data = data.split('\n')
for row in data:
    all_x.append(row.split()[0])
    all_y.append(row.split()[1])
    all_f.append(int(row.split()[2]))

print all_f

cap = cv2.VideoCapture(clip)
count = 0
dots = []

while (1):
    ret, frame = cap.read()

    if ret:
        for dot in dots:
            cv2.circle(frame, dot, 2, (0, 0, 255), thickness=-1)

        try:
            i = all_f.index(count)
            x = float(all_x[i])
            y = -float(all_y[i])
            dots.append((int(x), int(y)))

        except ValueError, e:
            pass
            
        count += 1
        cv2.imshow('Stream', frame)
        cv2.waitKey()
    else:
        break

cap.release()

#!/usr/local/bin/python

import sys
import cv2
import cv2.cv as cv
import numpy as np
import os.path
from time import sleep
import plotting as plot


# two arguments:
# 1. A video clip or image sequence
# 2. The the 2D trajectory corresponding to that clip

if len(sys.argv) < 3:
    print "Usage : python trace.py <image_sequence> <trajectory> <outfile>"
    sys.exit(0)

clip = sys.argv[1]
trajectory = sys.argv[2]

try:
    outfilename = sys.argv[3]
except:
    pass

t_dir = os.path.dirname(trajectory)
tracer_stats = t_dir + '/tracer_stats.txt'

if os.path.isdir(clip):
    print "> Input Type: Image Sequence"
    clip = clip + '/frame_%05d.png'
else:
    print "> Input Type: Video File"

# Get trajectory data
with open(trajectory) as datafile:
    data = datafile.read()
    datafile.close()

with open(tracer_stats) as datafile:
    stats = datafile.read()
    datafile.close()

stats = stats.split('\n')
avg_speed = stats[0]
distance = stats[1]

avg_speed = 'Average Speed: ' + str(avg_speed) + 'mph'
distance = 'Distance covered*: ' + str(distance) + 'm'

all_x = []
all_y = []
all_f = []

data = data.split('\n')
for row in data:
    all_x.append(row.split()[0])
    all_y.append(row.split()[1])
    all_f.append(int(row.split()[2]))

cap = cv2.VideoCapture(clip)
count = 0
dots = []
save = False

if outfilename is not None:
    save = True
    fps = 30.0
    capsize = (1280, 720)
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    vout = cv2.VideoWriter()
    success = vout.open(outfilename, fourcc, fps, capsize, True)

while (1):
    ret, frame = cap.read()

    if ret:
        prev = None

        cv2.putText(frame, avg_speed, (600, 575),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.5,
                    thickness=2,
                    color=(255, 255, 255))
        cv2.putText(frame, distance, (600, 630),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1.5,
                    thickness=2,
                    color=(255, 255, 255))

        for dot in dots:
            # draw the dot
            # cv2.circle(frame, dot, 4, (0, 0, 255), thickness=-1)

            # connect the dots
            if prev is not None:

                cv2.line(frame, prev, dot,
                         lineType=cv.CV_AA,
                         color=(0, 0, 255),
                         thickness=2)

            prev = dot

        try:
            i = all_f.index(count)
            x = float(all_x[i])
            y = -float(all_y[i])
            dots.append((int(x), int(y)))

        except ValueError, e:
            pass
        cv2.imshow('Stream', frame)
        cv2.waitKey(1)
        count += 1
        if save:
            vout.write(frame)

        
    else:
        break

cap.release()
if save:
    vout.release()
    vout = None
cv2.destroyAllWindows()

#!/usr/local/bin/python

import cv2
import cv2.cv as cv
import sys
import os.path

filename = sys.argv[1]

cap = cv2.VideoCapture(filename)

name, ext = os.path.splitext(filename)

outdir = name

if not os.path.exists(outdir):
    os.makedirs(outdir)

print "outdirectory:", outdir

count = 0

while (cap.isOpened()):

    ret, frame = cap.read()

    if ret is True:
        count += 1
        if count % 100 == 0:
            print "write frame:", count
        fname = str(count)
        fname = fname.zfill(5)
        outfilename = outdir + '/' + fname + '.png'
        cv2.imwrite(outfilename, frame)

    else:
        break

cap.release()
cv2.destroyAllWindows()

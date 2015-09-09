#!/usr/local/bin/python

''' postPoints.py <infile> <points outfile> <image_outfile>

User interface for marking up goal post corners. Input file is a video or image
sequence, of which the first frame is taken and displayed to the user, who is
prompted to click the bottom left, top left, top right and bottom right corners

points are saved to file, and the relevant frame is saved too.

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

# Colors and fonts
blue = (255, 0, 0)
red = (0, 0, 255)
white = (255, 255, 255)
font = cv2.FONT_HERSHEY_DUPLEX

# No optional arguments
if len(sys.argv) != 4:
    print "Usage : /postPoints.py <infile> <points outfile> <image_outfile>"
    sys.exit(0)

infilename = sys.argv[1]
outfilename = sys.argv[2]
image_outfile = sys.argv[3]

# supplied path can be a directory containing an image sequence: 00001.png
if os.path.isdir(infilename):
    infilename = infilename + '/frame_%05d.png'

outfile = open(outfilename, 'w')
counter = 0

cap = cv2.VideoCapture(infilename)

ret, original = cap.read()

# need a bunch of copies because we're going to draw all over them
frame0 = original.copy()
frame1 = original.copy()
frame2 = original.copy()
frame3 = original.copy()
frame4 = original.copy()
frame5 = original.copy()

print "Write image:", image_outfile
cv2.imwrite(image_outfile, original)

cv2.namedWindow('Click the Goalpost Corners')
cv2.setMouseCallback('Click the Goalpost Corners', click)


cv2.putText(frame0, "When prompted, click the corners of the goalposts.",
            (260, 450), font, 1, blue)


cv2.putText(frame0, "Press Q to quit, or R to restart at any time.",
            (260, 500), font, 1, blue)


cv2.putText(frame0, "Click anywhere to begin.",
            (260, 550), font, 1, blue)

cv2.putText(frame1, "Click the bottom left corner", (360, 600),
            fontFace=font,
            fontScale=1.2,
            color=white,
            thickness=2)

cv2.putText(frame2, "Click the top left corner", (360, 600),
            fontFace=font,
            fontScale=1.2,
            color=white,
            thickness=2)

cv2.putText(frame3, "Click the top right corner", (360, 600),
            fontFace=font,
            fontScale=1.2,
            color=white,
            thickness=2)

cv2.putText(frame4, "Click the bottom right corner", (360, 600),
            fontFace=font,
            fontScale=1.2,
            color=white,
            thickness=2)

cv2.putText(frame5, "Great! Press Q to exit.", (360, 600),
            fontFace=font,
            fontScale=1.2,
            color=white,
            thickness=2)

# Mouse click will iterate through the frame list
frames = (frame0, frame1, frame2, frame3, frame4, frame5)
current = frame0

while(1):
    cv2.imshow('Click the Goalpost Corners', current)
    if cv2.waitKey(20) & 0xFF == 113:
        break

cap.release()
cv2.destroyAllWindows()
outfile.close()

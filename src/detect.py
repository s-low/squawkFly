#!/usr/local/bin/python

# MAIN DETECTION TEST FILE
import sys
import cv2
import numpy as np
import os.path
sys.path.append('/usr/local/lib/python2.7/site-packages')

cap = 0
debugging = False
tracking = True
paused = False
point_index = 0

max_area = 1500
min_area = 250

outfile = None

startOfFile = True
time = 0

dots = []
detections = []

# default to showing the detection streams
view = True
try:
    if sys.argv[3] == 'suppress':
        view = False
except IndexError:
    pass


def main():
    global cap
    global outfile
    keys = {-1: cont, 116: track, 112: pause, 113: quit, 100: debug}

    if len(sys.argv) < 2:
        print "Usage : ./detect.py <image_sequence> *<outfile>* *<view>*"
        sys.exit(0)

    path = sys.argv[1]

    # supplied path can be a directory containing an image sequence: 00001.png
    if os.path.isdir(path):
        print "> INPUT: Image Sequence"
        path = path + '/frame_%05d.png'
    else:
        print "> INPUT: Video File"

    # otherwise just go and get the video file
    cap = cv2.VideoCapture(path)
    # print "Frame rate: " + repr(cap.get(5))

    outfile = open('data/data_detections.txt', 'w')

    # read three frames for initialisation
    ret, frame0 = cap.read()
    grayed0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)

    ret, frame1 = cap.read()
    grayed1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

    ret, frame2 = cap.read()
    grayed2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    # Get on with the capture
    while(cap.isOpened()):

        temp = frame1  # without contours
        current = diff(grayed0, grayed1, grayed2)

        current = morph(current)

        temp_thresh = current.copy()

        # Frame1 gets modified with contours
        if tracking:
            search(frame1, temp_thresh)
            # for dot in dots:
            #     cv2.circle(frame1, dot, 3, (0, 0, 255), thickness=-1)

        if view:
            cv2.imshow('Feed', frame1)

            if debugging:
                cv2.imshow('Threshold Image', current)  # why does this go odd
            else:
                cv2.destroyWindow('Threshold Image')

        if paused:
            key = cv2.waitKey()

            if key == 112:
                pause()

            if key == 113:
                cap.release()
                cv2.destroyAllWindows()
                outfile.close()
                sys.exit()

        # Next iteration
        ret, next_frame = cap.read()

        if ret is True:
            frame0 = temp
            grayed0 = grayed1

            frame1 = frame2
            grayed1 = grayed2

            frame2 = next_frame
            grayed2 = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

            try:
                keys[cv2.waitKey(1)]()
            except KeyError:
                continue
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    outfile.close()


# returns a thresholded difference image
def diff(f0, f1, f2):
    d1 = cv2.absdiff(f2, f1)

    d2 = cv2.absdiff(f1, f0)

    overlap = cv2.bitwise_and(d1, d2)

    # binary threshold(src, thresh, maxval, type)
    ret, thresh = cv2.threshold(overlap, 40, 255, cv2.THRESH_BINARY)
    return thresh


# returns a re-thresholded image after blur and open/close/erode/dilate
def morph(image):
    kernel = np.ones((11, 11), np.uint8)
    image = cv2.dilate(image, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    ret, image = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
    return image


# search a frame for candidates
def search(src, thresh):
    global detections
    global point_index
    global outfile
    global startOfFile
    global time
    time += 1
    objectDetected = False

    # find contours in threshold
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        objectDetected = True
    else:
        objectDetected = False

    # draw the contours onto the source image
    if objectDetected:
        cv2.drawContours(src, contours, -1, (0, 255, 0), 3)
        for contour in contours:

            area = cv2.contourArea(contour)

            # filter by size
            if area < max_area and area > min_area:
                # filter by squareness
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(src, str(area), (x, y), cv2.FONT_HERSHEY_PLAIN,
                            0.8, (255, 255, 255))
                if square(h, w) and circular(area, h, w):
                    point_index += 1
                    cv2.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    cx = x + float(w) / 2.0
                    cy = -1 * (y + float(h) / 2.0)
                    dots.append((int(cx), -int(cy)))

                    d = (cx, cy, time, point_index)
                    detections.append(d)

                    if not startOfFile:
                        outfile.write('\n')

                    outfile.write(repr(cx) + ' ' + repr(cy) + ' ' +
                                  repr(time) + ' ' + repr(point_index))

                    if startOfFile is True:
                        startOfFile = False


def square(h, w):
    shorter = min((h, w))
    longer = max((h, w))
    squareness = abs((float(longer) / float(shorter)) - 1)
    if squareness < 0.5:
        return True
    else:
        return False


# if perfectly circular then ration of areas: contour/box = pi/4
def circular(area, h, w):
    ratio = float(area) / (float(h) * float(w))
    pi4 = (3.142 / 4.0)
    closeness = abs(ratio - pi4)

    if closeness < 0.4:
        return True
    else:
        return False


def track():
    global tracking
    tracking = not tracking

    if tracking:
        print "Tracking ON"
    else:
        print "Tracking OFF"


def pause():
    global paused
    paused = not paused


def debug():
    global debugging
    debugging = not debugging

    if debugging:
        print "Debug ON"
    else:
        print "Debug OFF"


def cont():
    return


def quit():
    global cap
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Procedural body
main()

# optionally specify a dedicated outfile
try:
    outfile = open(sys.argv[2], 'w')

    for d in detections:
        outfile.write(str(d[0]) + ' ' + str(d[1]) + ' ' +
                      str(d[2]) + ' ' + str(d[3]) + '\n')

    outfile.close()

except IndexError:
    pass

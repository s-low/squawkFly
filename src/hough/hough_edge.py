# HOUGH CIRCLES - EDGE DETECTED VERSION

import cv2
import numpy as np
import sys

def diff(f0, f1, f2):
	d1 = cv2.absdiff(f2, f1)
	d2 = cv2.absdiff(f1, f0)
	overlap = cv2.bitwise_and(d1, d2)

	# binary threshold(src, thresh, maxval, type)
	ret, overlap = cv2.threshold(overlap, 30, 255, cv2.THRESH_BINARY)
	return overlap

if len(sys.argv)!=2:                 
    print "Usage : python detect.py <video_file>"
    sys.exit(0)

cap = cv2.VideoCapture(sys.argv[1])

ret, frame0 = cap.read()
ret, frame1 = cap.read()
ret, frame2 = cap.read()

grayed0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
grayed1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
grayed2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

# Get on with the capture
while(cap.isOpened()):
	
	current = diff(grayed0, grayed1, grayed2)


	# MORPHOLOGICAL OPERATIONS
	kernel = np.ones((5,5),np.uint8)
	# current = cv2.dilate(current, kernel)
	current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel)
	# current = cv2.GaussianBlur(current, (3,3), 0)

	current = cv2.Canny(current, 200, 100)

	# HOUGH CIRCLE DETECTION
	dp = 12
	minDist = 40
	maxRad = 30
	minRad = 1

	circles = cv2.HoughCircles(current, cv2.cv.CV_HOUGH_GRADIENT, dp, minDist, minRadius = minRad, maxRadius = maxRad)
	colored = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)

	if circles is not None:
		circles = circles[0]
		
		for i in range(0, len(circles)):
			circle = circles[i]
			x = int(circle[0])
			y = int(circle[1])
			radius = int(circle[2])
			cv2.circle(colored, (x, y), radius, (0,0,255), thickness=2)
	
	cv2.imshow('Difference', colored)

	# Next iteration
	ret, next_frame = cap.read()
	if ret == True:
		grayed0 = grayed1
		grayed1 = grayed2
		grayed2 = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

		# hit q to quit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()

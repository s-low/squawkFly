# MAIN DETECTION TEST FILE

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
	# current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel)
	# current = cv2.GaussianBlur(current, (3,3), 0)
	
	cv2.imshow('Difference', current)

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

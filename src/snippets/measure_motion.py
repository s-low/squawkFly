import cv2
import numpy
import sys

# total is a rough metric for motion in a clip
# sum of pixel values in differenced frames
total = 0

def diff(f0, f1, f2):
	global total
	d1 = cv2.absdiff(f2, f1)
	d2 = cv2.absdiff(f1, f0)
	overlap = cv2.bitwise_and(d1, d2)

	# binary threshold(src, thresh, maxval, type)
	# changing thresh val will affect total
	ret, thresh = cv2.threshold(overlap, 20, 255, cv2.THRESH_BINARY)
	total = total + cv2.sumElems(thresh)[0]
	return thresh

if len(sys.argv)!=2:                 
    print "Usage : python mar1.py <video_file>"
    sys.exit(0)

cap = cv2.VideoCapture(sys.argv[1])

# gobble the intro - 70 frames/2sec+change roughly covers either MAR version
for x in xrange(1,70):
	ret, frame1 = cap.read()

ret, frame0 = cap.read()
ret, frame1 = cap.read()
ret, frame2 = cap.read()

grayed0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
grayed1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
grayed2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

while(cap.isOpened()):

	cv2.imshow('Difference', diff(grayed0, grayed1, grayed2))

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

print '%g' % total
cap.release()
cv2.destroyAllWindows()


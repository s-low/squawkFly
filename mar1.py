import cv2
import numpy
import sys

total = 0

def diffImg(t1, t2, t3):
	global total
	d1 = cv2.absdiff(t3,t2)
	d2 = cv2.absdiff(t2,t1)
	diff = cv2.bitwise_and(d1, d2)

	# threshold(src, thresh, maxval, type)
	ret, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
	total = total + cv2.sumElems(thresh)[0]
	return thresh

if len(sys.argv)!=2:                 
    print "Usage : python mar1.py <video_file>"
    sys.exit(0)

cap = cv2.VideoCapture(sys.argv[1])

# gobble the intro
for x in xrange(1,70):
	ret, frame1 = cap.read()

ret, frame1 = cap.read()
ret, frame2 = cap.read()
ret, frame3 = cap.read()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
gray3 = cv2.cvtColor(frame3, cv2.COLOR_RGB2GRAY)

while(cap.isOpened()):

	cv2.imshow('Difference', diffImg(gray1, gray2, gray3))
	ret, frame3 = cap.read()
	
	if ret == True:
		gray1 = gray2
		gray2 = gray3
		gray3 = cv2.cvtColor(frame3, cv2.COLOR_RGB2GRAY)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

print '%g' % total
cap.release()
cv2.destroyAllWindows()


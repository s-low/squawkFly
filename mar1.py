import cv2
import numpy

def diffImg(t1, t2, t3):
	d1 = cv2.absdiff(t3,t2)
	d2 = cv2.absdiff(t2,t1)
	return cv2.bitwise_and(d1, d2)

cap = cv2.VideoCapture("fk_test_1.mp4")

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

	


cap.release()
cv2.destroyAllWindows()


import cv2
import numpy

cap = cv2.VideoCapture("fk_test_1.mp4")

# Define the codec and create writer from it
fps = 30.0
capsize = (1280,720)
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case

vout = cv2.VideoWriter()
success = vout.open('output.mov',fourcc,fps,capsize,True) 

while(cap.isOpened()):
	ret, frame = cap.read()

	if ret == True:
		vout.write(frame) 
		cv2.imshow('frame',frame)
	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else: 
		break

cap.release()
vout.release()
vout = None
cv2.destroyAllWindows()

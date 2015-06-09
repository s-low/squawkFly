# MAIN DETECTION TEST FILE
import cv2
import numpy as np
import sys

cap = 0
debugging = False
tracking = False
paused = False

def main():
	global cap
	keys = {-1: cont, 116 : track, 112 : pause, 113: quit, 100: debug}

	if len(sys.argv)!=2:                 
		print "Usage : python detect.py <video_file>"
		sys.exit(0)

	cap = cv2.VideoCapture(sys.argv[1])

	# read three frames for initialisation
	ret, frame0 = cap.read()
	grayed0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
	
	ret, frame1 = cap.read()
	grayed1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
	
	ret, frame2 = cap.read()
	grayed2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)	

	# Get on with the capture
	while(cap.isOpened()):
		
		current = diff(grayed0, grayed1, grayed2)
		current = morph(current)
		

		if tracking:
			print "tracking"

		cv2.imshow('Feed', frame1)

		if debugging:
			cv2.imshow('difference', current)
		else:
			cv2.destroyWindow('difference')

		if paused:
			cv2.waitKey()
			pause()

		# Next iteration
		ret, next_frame = cap.read()

		if ret == True:
			frame0 = frame1
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
		
# returns a thresholded difference image
def diff(f0, f1, f2):
	d1 = cv2.absdiff(f2, f1)
	d2 = cv2.absdiff(f1, f0)
	overlap = cv2.bitwise_and(d1, d2)

	# binary threshold(src, thresh, maxval, type)
	ret, overlap = cv2.threshold(overlap, 40, 255, cv2.THRESH_BINARY)
	return overlap

# returns a re-thresholded image after blur and open/close/erode/dilate
def morph(image):
	# kernel = np.ones((7,7),np.uint8)
	# image = cv2.dilate(image, kernel)
	# image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
	image = cv2.GaussianBlur(image, (15,15), 0)
	ret, image = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
	return image

def search(src):
	contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

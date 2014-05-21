'''
Created on 09.04.2014

@author: tabuchte
'''

import numpy as np
import cv2
import webcam

# refImg, refImg_hsv = webcam.captureRefImage()
cap = cv2.VideoCapture('c:/Users/tabuchte/coding/matlab/mean_shift/ex4/tabletennis.avi')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window 
r,h,c,w = 27,31,136,28  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

cv2.imshow('roi', roi)
cv2.imshow('hsv_roi', hsv_roi)
cv2.waitKey(0)

mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    # frame, frame_hsv = webcam.get_frame()
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    print dst

    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # Draw it on image
    x,y,w,h = track_window
    cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
    cv2.imshow('img2',frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
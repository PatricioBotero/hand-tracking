'''
Created on 02.04.2014

@author: tabuchte
'''

import cv2


camera = cv2.VideoCapture(0)

def captureRefImage():
    for i in range(0,30):
        retval, cameraCapture = camera.read()
        cv2.imshow('test', cameraCapture)
        cv2.waitKey(10)

    retval, cameraCapture = camera.read()
    cv2.imshow('test', cameraCapture)
    cv2.waitKey(10)
    
    cameraCapture_hsv = cv2.cvtColor(cameraCapture, cv2.COLOR_BGR2HSV)
    
    return cameraCapture, cameraCapture_hsv

def get_frame():
    retval, frame = camera.read()
    framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frame, framehsv
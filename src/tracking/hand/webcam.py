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
    
    return cameraCapture
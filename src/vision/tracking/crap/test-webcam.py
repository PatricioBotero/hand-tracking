'''
Created on 05.04.2014

@author: Patrick
'''

import cv2

camera = cv2.VideoCapture(0)
retval, image = camera.read()
print(image)
cv2.imshow('asdf', image)
cv2.waitKey()
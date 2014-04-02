'''
Created on 02.04.2014

@author: tabuchte
'''

from numpy import *
from numpy.random import *
import cv2

def computeHist(img, center, bbsize):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = zeros(img2.shape[:2], uint8)
    mask[center[0]-bbsize:center[0]+bbsize-1, center[1]-bbsize:center[1]+bbsize-1] = 255
    hist1 = cv2.calcHist([img2], [0], mask, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [1], mask, [256], [0, 256])
    hist3 = cv2.calcHist([img2], [2], mask, [256], [0, 256])
    return concatenate((hist1, hist2, hist3), axis=0)



def getRefHistogram(refImage, bbsize, center):
    hist = computeHist(refImage, center, bbsize)
    return hist
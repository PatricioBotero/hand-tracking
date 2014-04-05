'''
Created on 02.04.2014

@author: tabuchte
'''

from numpy import *
from numpy.random import *
import cv2

def computeHist(img, center, bbsize):
    mask = zeros(img.shape[:2], uint8)
    mask[center[0]-bbsize:center[0]+bbsize-1, center[1]-bbsize:center[1]+bbsize-1] = 255
    hist1 = cv2.calcHist([img], [0], mask, [256], [0, 256])
    hist2 = cv2.calcHist([img], [1], mask, [256], [0, 256])
    hist3 = cv2.calcHist([img], [2], mask, [256], [0, 256])
    return concatenate((hist1, hist2, hist3), axis=0)

def getRefHistogram(refImage, bbsize, center):
    hist = computeHist(refImage, center, bbsize)
    return hist

def normalizeVector(vec):
    sumWeights = sum(vec)
    for i in range(len(vec)):
        vec[i] = vec[i] / sumWeights
        
def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]     # cumsum
    u0, j = random(), 0
    for u in [(u0+i)/n for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j-1)
    return indices

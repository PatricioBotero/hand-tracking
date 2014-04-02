'''
Created on 02.04.2014

@author: tabuchte
'''

from numpy import *
from numpy.random import *
from pylab import *
from itertools import izip
import time
import cv2

from tracking.hand.histogram_operations import *
from tracking.hand.particle_filter import *
from tracking.hand.webcam import *

dt = 1;
updateMatrix = np.array([[1,0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
path2 = "C:/Users/tabuchte/coding/python/test.jpg"

center = array([240, 280])
bbsize = 15

Xstd_pos = 40; 
Xstd_vec = 20; 
N = 250; 




refImage = captureRefImage()
refHist = getRefHistogram(refImage, bbsize, center)
dim_y, dim_x = refImage.shape[:2]

X = np.vstack( (np.random.random_integers(bbsize, dim_y-bbsize, size=(1.,N)), 
    np.random.random_integers(bbsize, dim_x-bbsize, size=(1.,N)), 
    np.zeros([2, N])) )
# --> convert an arry into np.matrix

while True:
    xUpdate = np.dot(updateMatrix, X)
    X[:2,:] = X[:2,:] + Xstd_pos * (np.random.random_sample((2, N))*2-1)
    X[2:4,:] = X[2:4,:] + Xstd_vec * (np.random.random_sample((2, N))*2-1)

    retval, frame = camera.read()
    framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    weights = []
    for p in range(N):

        if (X[1,p] > bbsize) & (X[1,p] < dim_y-bbsize) & (X[2,p] > bbsize) & (X[2,p] < dim_x-bbsize):
        
            #weight = np.exp(-np.sqrt( pow(X[0,p]-center[0],2) + pow(X[1,p]-center[1],2) ) )
            
            hist = computeHist(framehsv, X[:2,p], bbsize)
            
            bdist = sum(sqrt(refHist * hist))
            weight = 100-(bdist)
            
            weights.append(weight)
            
        else:
            weights.append(0)
            
        ''' show particle '''
        cv2.circle(frame, (np.int(X[1,p]), np.int(X[0,p])), 4, (0,255,0), -1)
        print ('velocity ', X[2:4,p])
         
    # drawing
    cv2.imshow('test', frame)
    cv2.waitKey(30)
    
    # resampling
    sumWeights = np.sum(weights)
    for i in range(N):
        weights[i] = weights[i] / sumWeights
    indices = resample(weights)

    for i in range(len(indices)):
        X[:, i] = X[:, indices[i]] # maybe i+1
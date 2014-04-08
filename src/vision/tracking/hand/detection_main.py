'''
Created on 02.04.2014

@author: tabuchte
'''

from numpy import *
from numpy.random import *
from pylab import *
from itertools import izip
import time


from MathUtils import *
from particle_filter import *
import webcam

def showParticles(X, frame, waittime, N):
    for p in range(N):
        ''' show particle '''
        cv2.circle(frame, (np.int(X[1,p]), np.int(X[0,p])), 1, (0,255,0), -1)
        print ('velocity ', X[2:4,p])
    cv2.imshow('test', frame)
    cv2.waitKey(waittime)

''' settings -> json'''
dt = 1;
updateMatrix = np.array([[1,0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

center = array([240, 280])
bbsize = 15
Xstd_pos = 40 
Xstd_vec = 20 
N = 250
waittime = 30

''' refHist '''
refImage, refImage_hsv = webcam.captureRefImage()
refHist = getRefHistogram(refImage_hsv, bbsize, center)
dim_y, dim_x = refImage_hsv.shape[:2]



X = initParticles(bbsize, dim_y, dim_x, N)

while True:
    updateParticles(updateMatrix, Xstd_pos, Xstd_vec, X, N)
    frame, framehsv = webcam.get_frame()
    weights = computeWeights(X, bbsize, dim_y, dim_x, refHist, framehsv, N)
    showParticles(X, frame, waittime, N)
    normalizeVector(weights)
    indices = resample(weights)

    for i in range(len(indices)):
        X[:, i] = X[:, indices[i]] # maybe i+1
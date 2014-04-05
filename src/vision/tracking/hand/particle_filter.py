'''
Created on 30.03.2014

@author: tabuchte
'''
from numpy import *
from numpy.random import *
from pylab import *
from MathUtils import *

def initParticles(bbsize, dim_y, dim_x, N):
    X = vstack( (np.random.random_integers(bbsize, dim_y-bbsize, size=(1.,N)), 
    np.random.random_integers(bbsize, dim_x-bbsize, size=(1.,N)), 
    zeros([2, N])) )
    return X
    
def updateParticles(updateMatrix, Xstd_pos, Xstd_vec, X, N):
    xUpdate = np.dot(updateMatrix, X)
    X[:2,:] = X[:2,:] + Xstd_pos * (np.random.random_sample((2, N))*2-1)
    X[2:4,:] = X[2:4,:] + Xstd_vec * (np.random.random_sample((2, N))*2-1)
    
def computeLikelihood(refHist, hist):
    return 100 - sum(sqrt(refHist * hist))

def computeWeights(X, bbsize, dim_y, dim_x, refHist, framehsv, N):
    weights = []
    for p in range(N):
        if (X[1,p] > bbsize) & (X[1,p] < dim_y-bbsize) & (X[2,p] > bbsize) & (X[2,p] < dim_x-bbsize):
            hist = computeHist(framehsv, X[:2,p], bbsize)
            weight = computeLikelihood(refHist, hist)
            weights.append(weight)
        else:
            weights.append(0)
    return weights
'''
Created on 08.04.2014

@author: tabuchte
'''

import numpy as np
import tracking.hand.webcam as webcam

def getBinnedRGB(img, i, j, redBins, greenBins, blueBins):
    R = np.floor(img[i,j,0] / (256/redBins));
    G = np.floor(img[i,j,1] / (256/greenBins));
    B = np.floor(img[i,j,2] / (256/blueBins));
    return R, G, B

def grayPDF(img, center, w_halfsize, bins, maxVal):
    hist = np.zeros([bins])    
    rmin, rmax, cmin, cmax = getBoundingBox(center, w_halfsize)
    maxdist = pow(rmin-center[0],2) + pow(cmin-center[1],2) + 1
    
    for i in range(rmin, rmax):
        for j in range(cmin, cmax):
            # this is my kernel --> todo: use appropriate kernel
            dist = pow(i-center[0],2) + pow(j-center[1],2) + 1
            w = maxdist - dist
            G = np.floor(img[i,j] / (maxVal/bins));
            hist[G] = hist[G] + 1;
    
    # normalize
    return hist / np.sum(hist)

def rgbPDF(img, center, w_halfsize, redBins, greenBins, blueBins):
    
    hist_3d = np.zeros([redBins, greenBins, blueBins])    
    rmin, rmax, cmin, cmax = getBoundingBox(center, w_halfsize)
    maxdist = pow(rmin-center[0],2) + pow(cmin-center[1],2) + 1
    
    for i in range(rmin, rmax):
        for j in range(cmin, cmax):
            # this is my kernel --> todo: use appropriate kernel
            dist = pow(i-center[0],2) + pow(j-center[1],2) + 1
            w = maxdist - dist
            R, G, B = getBinnedRGB(img, i, j, redBins, greenBins, blueBins)
            hist_3d[R,G,B] = hist_3d[R,G,B] + 1;
    
    return computeNormalizedHistFrom3dHist(hist_3d, redBins, greenBins, blueBins)
    
def computeNormalizedHistFrom3dHist(hist_3d, redBins, greenBins, blueBins):
    hist_1d = np.zeros([redBins*greenBins*blueBins])
    sum_bins = 0
    
    for i in range(0,redBins-1):
        for j in range(0,greenBins-1):
            for k in range(0,blueBins-1):
                index = i*greenBins*blueBins+j*blueBins+k
                hist_1d[index] = hist_3d[i,j,k]
                sum_bins = sum_bins + hist_1d[index]
                
    # normalize
    hist_1d = hist_1d / sum_bins
    return hist_1d
    
def getBoundingBox(center, w_halfsize):
    rmin = center[0] - w_halfsize[0]
    rmax = center[0] + w_halfsize[0]
    cmin = center[1] - w_halfsize[1]
    cmax = center[1] + w_halfsize[1]
    return rmin, rmax, cmin, cmax
    
def computeBhattacharyyaCoef(q,p):
    coef = 0
    for i in range(0,len(q)-1):
        coef += np.sqrt(q[i]*p[i])
    return coef
    
def meanShiftWebcam():
    
    # initialize
    refImg, refImg_hsv = webcam.captureRefImage()
    height = refImg.shape[0]
    width = refImg.shape[1]
    w_halfsize = [30, 30]
    redBins = 8; greenBins = 8; blueBins = 8;
    y0 = [np.round(height/2), np.round(width/2)]
    minDist = 0.1
    maxIterNum = 15
    
    # compute q
    
    q = rgbPDF(refImg, y0, w_halfsize, redBins, greenBins, blueBins)
    
    # repeat until the end of the sequence
    while 1:
    
        frame, framehsv = webcam.get_frame()
        rmin, rmax, cmin, cmax = getBoundingBox(y0, w_halfsize)
    
        # initialize estimated center y1 = y0
        y1 = y0
        
        # loop until d < epsilon
        M_length = minDist
        w = np.zeros([(rmax-rmin+1)*(rmax-rmin+1)])
        x = np.zeros([2, (rmax-rmin+1)*(rmax-rmin+1) ])
        for iter_index in range(0,maxIterNum):
        
            # calculate p
            p = rgbPDF(frame, y1, w_halfsize, redBins, greenBins, blueBins)
             
            # calculate w
            n = 1
            for i in range(rmin, rmax):
                for j in range(cmin, cmax):
                    R,G,B = getBinnedRGB(frame, i, j, redBins, greenBins, blueBins)
                    
                    index = int(R*greenBins*blueBins+G*blueBins+B)
                    if p[index] > 0:
                        w[n] = np.sqrt(q[index] / p[index])
                    
                    x[0,n] = i; x[1,n] = j;
                    n = n+1
            
            # estimate new target center y1
            y1 = np.dot(x,w) / np.sum(w)
            M_length = np.sqrt(pow(y1[0]-y0[0],2) + pow(y1[1]-y0[1],2))
            if M_length < minDist:
                break
            
            y1 = [int(y1[0]), int(y1[1])]
        
        # update target center
        y0 = y1
        print y1
    
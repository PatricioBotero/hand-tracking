'''
Created on 24.03.2014

@author: tabuchte
'''



from Tkinter import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import copy



def computeHist(img, center, bbsize):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[center[0]-bbsize:center[0]+bbsize-1, center[1]-bbsize:center[1]+bbsize-1] = 255
    hist1 = cv2.calcHist([img], [0], mask, [256], [0, 256])
    hist2 = cv2.calcHist([img], [1], mask, [256], [0, 256])
    # hist3 = cv2.calcHist([img], [2], mask, [256], [0, 256])
    return np.concatenate((hist1, hist2), axis=0)

def showImage(path):
    img = cv2.imread(path)
    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)

def getImage():
    retval, im = camera.read()
    return im

def captureRefImage(savePath):
    for i in range(0,30):
        cameraCapture = getImage()
        cv2.imshow('test', cameraCapture)
        cv2.waitKey(10)

    cameraCapture = getImage()
    cv2.imshow('test', cameraCapture)
    cv2.waitKey(10)
    
    cv2.imwrite(savePath, cameraCapture)

def getRefHistogram(filePath, bbsize, center):
    img = cv2.imread(filePath)
    hist = computeHist(img, center, bbsize)
    return img, hist

class Particle(object):
    pos = [0,0]
    vel = [0,0]
    weight = 0
    distance = 0
    
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel
        
    def getState(self):
        return np.matrix([self.pos[0], self.pos[1], self.vel[0], self.vel[1]]).T
    
    def move(self, dt, updateMatrix, stdPos, stdVel):
        xUpdate = updateMatrix * self.getState()
        self.pos[0] = xUpdate[0,0] + stdPos * np.random.normal()
        self.pos[1] = xUpdate[1,0] + stdPos * np.random.normal()
        self.vel[0] = xUpdate[2,0] + stdVel * np.random.normal()
        self.vel[1] = xUpdate[3,0] + stdVel * np.random.normal()
        
    def updateWeightBhattacharyya(self, refHist, frame, bbsize, hist):
        if (self.pos > [bbsize+1, bbsize+1]) & (self.pos < [frame.shape[0]-bbsize-1, frame.shape[1]-bbsize-1]):
            hist = computeHist(frame, self.pos, bbsize)
            bdist = 0.0
            for i in range(len(hist)):
                bdist += np.sqrt(hist[i,0]*refHist[i,0])
            self.weight = 100-(bdist)
        else:
            self.weight = 0
        
        
    def updateWeightDistance(self, center):
        self.distance = np.sqrt( pow(self.pos[0]-center[0],2) + pow(self.pos[1]-center[1],2) )
        self.weight = np.exp(self.distance)
        
def createParticle(pos, vel):
    return Particle(pos,vel)

def initializeParticles(frame, nparticles, bbsize):
    particles = []
    for i in range(nparticles):
        m, n = frame.shape[:2]
        y = random.randint(bbsize, m-2*bbsize)
        x = random.randint(bbsize, n-2*bbsize)
        pos = [y,x]
        particles.append(createParticle(pos, [0,0]))
    return particles

def resampleParticles(particles, weights):
    cumsum = np.cumsum(weights)
    particles2 = []
    for i in range(len(particles)):
        r = random.random()
        for j in range(len(cumsum)):
            if r > cumsum[j]:
                pwin = particles[j]
                particles2.append(createParticle(pwin.pos, pwin.vel))
                break
    return particles2

        
def getFrame(path):
    # return HistUtils.getImage()
    return cv2.imread(path2)
     
def debug(particles, center, weights):
    
    print('------ center: ', center)
    print('------ sum weights: ', np.sum(weights))
    print('------ nparticles: ', len(particles))
    
    for i in range(len(particles)):
        p = particles[i]
        print('particle ', i, ' - pos - ', p.pos, ' - dist - ', p.distance, ' - vel - ', p.vel, ' - weight - ', p.weight)
    
        
if __name__ == '__main__':
    
    cameraPort = 0
    center = [240, 280]
    bbsize = 15
    nparticles = 10
    stdPos = 0.5
    stdVel = 0.5
    dt = 1
    updateMatrix = np.matrix("1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1")
    path2 = "C:/Users/tabuchte/coding/python/test.jpg"
    
    camera = cv2.VideoCapture(cameraPort)
    img, refHist = getRefHistogram(path2, bbsize, center)
    particles = initializeParticles(img, nparticles, bbsize)
    
    while True:
        
        weights = []
        frame = getFrame(path2)
        
        
        cv2.circle(frame, (np.int(center[1]), np.int(center[0])), 4, (0,0,255), -1) # x,y
        
        for i in range(len(particles)):
        
            p = particles[i]
            p.move(dt, updateMatrix, stdVel, stdPos)
            p.updateWeightDistance(center)
            
            weights.append(p.weight)
        
            ''' show particle '''
            cv2.circle(frame, (np.int(p.pos[1]), np.int(p.pos[0])), 4, (0,255,0), -1)
            
        ''' normalize weights '''
        sumWeights = np.sum(weights)
        for i in range(len(particles)):
            weights[i] = weights[i] / sumWeights  
        
        
        debug(particles, center, weights)
        
            
        cv2.imshow('test', frame)
        cv2.waitKey(0)
        
        particles = resampleParticles(particles, weights)
        
        


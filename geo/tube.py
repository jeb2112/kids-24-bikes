from operator import eq
from ssl import PROTOCOL_TLSv1_1
import numpy as np
import cv2
import matplotlib.pyplot as plt
# pip install peakutils rolls back numpy, which messes up other things.
# import peakutils
import copy
# relative intra-package import. can use .misc for module in the same directory as this file,
# or geo.misc as well, resolves to the same thing.
from .misc import *
from geo.display import Display

############
# class Tube
############

class Tube():
    def __init__(self):
        # pt1 is leftmost point, pt2 rightmost
        self.pt1 = np.array([0,0])
        self.pt2 = np.array([0,0])
        self.m = 0
        self.b = 0
        self.A = 0
        self.len = 0
        self.rho = 0
        self.theta = 0
        self.fixed = None

    def setpts(self,pt1,pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self.m,self.b = pts2eq(pt1,pt2)
        self.A = np.arctan(self.m)
        # list() unarrays the np.array, while [] retains it
        self.rho,self.theta = coord2angle([np.concatenate((pt1,pt2),axis=0)])
        self.len = self.l()

    def seteqn(self,m,b):
        self.m = m
        self.b = b
        if self.pt2[0]==0:
            # had cols,rows,mmpx as globals previously in a single file. still awkward
            # self.pt2[0]=cols
            self.pt2[0] = 100
        self.pt1,self.pt2 = eq2pts((m,b),(self.pt1[0],self.pt2[0]))
        self.pt1 = np.array(self.pt1)
        self.pt2 = np.array(self.pt2)
        self.A = np.arctan(self.m)
        self.rho,self.theta = coord2angle([np.concatenate((self.pt1,self.pt2),axis=0)])
        self.len = self.l()

    def setrhotheta(self,rho,theta):
        self.rho = rho
        self.theta = theta
        self.pt1,self.pt2 = angle2coord([(self.rho,self.theta)])
        self.m,self.b = pts2eq(self.pt1,self.pt2)
        self.A = np.arctan(self.m)
        self.len = self.l()

    def settheta(self,theta):
        if self.fixed is None:
            print('Fixed point not initialized')
            return
        else:
            m = np.tan(theta-np.pi/2)
            b = m * (-self.fixed[0]) + self.fixed[1]
            self.seteqn(m,b)

    def l(self):
        return(np.sqrt(pow(self.pt1[0]-self.pt2[0],2) + pow(self.pt1[1]-self.pt2[1],2)))
    
    def y(self,x):
        return(self.m * x + self.b)

    def x(self,y):
        return( (y - self.b) / self.m)


class Fork(Tube):
    def __init__(self,mmpx=None):
        self.type = 'suspension'
        self.axle2crown = 0
        self.offset = 0
        if mmpx is not None:
            self.cv = Convert(mmpx=mmpx)

    def calcOffset(self,headtube):
        # convention is pt1 is to the left of pt2
        offsetAngle = np.arcsin( (self.pt2[0]-self.pt1[0]) / self.axle2crown) + (headtube.A-np.pi/2)
        self.offset = self.axle2crown * np.sin(offsetAngle)
        print(('offset = %.1f cm' % self.cv.PX2CM(self.offset)))


# base class for a tubeset, part of an overall geometry
# extend with methods for image processing
class Tubeset():
    def __init__(self,mmpx=None):
        self.tubes = dict(dt=Tube(),tt=Tube(),st=Tube(),ht=Tube(),cs=Tube(),ss=Tube())
        if mmpx is not None:
            self.cv = Convert(mmpx=mmpx)
            self.D = Display(mmpx=mmpx)

    def plotTubes(self,aimg,linew=2):
        rows,cols = np.shape(aimg)[0:2]
        # plot raw lines detection
        for tube in ['ht','st','dt','tt']:
            # for some reason can't follow this syntax with the line array as did with tuples
            #for x1,y1,x2,y2 in np.nditer(Lline):
            #    cv2.line(aimg2,(x1,y1),(x2,y2),(0,0,255),2)
            pt1 = np.array([0,self.tubes[tube].y(0)])
            pt2 = np.array([cols,self.tubes[tube].y(cols)])
            # cv2.line(aimg,tuple(self.tubes[tube].pt1.astype(int)),tuple(self.tubes[tube].pt2.astype(int)),(255,0,0),linew)
            cv2.line(aimg,tuple(pt1.astype(int)),tuple(pt2.astype(int)),(255,0,0),linew)
        self.D.plotfig(aimg)



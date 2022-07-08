from operator import eq
import os
from ssl import PROTOCOL_TLSv1_1
import sys
import io
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
import peakutils
import scipy.interpolate
import copy
import argparse
import re
import scipy.optimize
from geo.tube import Tube,Tubeset

figNo = 1
rows = 0 
cols = 0
mmpx = 0
def CM2PX(cm):
    return(int(np.round(cm/(mmpx/10))))
def PX2CM(px):
    return(px * mmpx/10.)
def PX2MM(px):
    return(px * mmpx)
def DEG2RAD(d):
    return(d * np.pi/180)
def RAD2DEG(r):
    return(r * 180./np.pi)

#################
# general methods
#################

def pts2eq(pt1,pt2):
    x1,y1 = pt1
    x2,y2 = pt2
    if x1 != x2:
        m = float(y2-y1)/float(x2-x1)
        b = y1 - m * x1
    else:
        m = float('NaN')
        b = float('NaN')
    return [m,b]

# this should probably return array not list
def eq2pts(eq,pt):
    m,b = eq
    x1,x2 = pt
    y1 = m*x1 + b
    y2 = m*x2 + b
    return ([x1,y1],[x2,y2])

def plotFig(img,blockFlag=False,cmap=None,title=None,figNum=None):
    global figNo
    if figNum is None:
        figNum = figNo
        figNo += 1

    plt.figure(figNum)
    if title is not None:
        plt.title(title)
    if cmap is not None:
        plt.imshow(img,cmap=cmap)
    else:
        plt.imshow(img)
    # not sure of usage of plt.ion(). it may be contra-indicated by show()?
    # plt.ion()
    # plt.show()
    # plt.pause(.001)
    # input('press to continue')
    plt.show(block=blockFlag)
    # figNo += 1

# CV graphics do work with pause for keypad that way plt.ion() was supposed to
def plotFigCV(img,title="Fig"):
    # this should work but doesn't
    # cv2.startWindowThread()
    cv2.namedWindow(title)
    cv2.imshow(title,img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def plotLines(bw,lines,blockFlag=False,cmap=None,title=None, figNum=None, color=(0,0,255)):
    # plot raw lines detection
    for line in lines:
        for x1,y1,x2,y2 in line:
            if cmap == "gray":
                cv2.line(bw,(x1,y1),(x2,y2),140,CM2PX(0.1))
            else:
                cv2.line(bw,(x1,y1),(x2,y2),color,CM2PX(0.1))
    plotFig(bw,blockFlag,cmap=cmap,title=title, figNum=figNum)

def coord2angle(line):
    for x1,y1,x2,y2 in line:
        if x1 != x2:
            m = float(y2-y1) / float(x2-x1)
            b = y1 - m*x1
            theta = (np.pi/2  + np.arctan(m)) - np.pi
            rho = -b / (m*np.cos(theta) - np.sin(theta)) 
        else:
            rho = float('NaN')
            theta = float('NaN')
        return(rho,theta)
        
def angle2coord(line):
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = float(x0 + 1000*(-b))
        y1 = float(y0 + 1000*(a))
        x2 = float(x0 - 1000*(-b))
        y2 = float(y0 - 1000*(a))
    return[[x2,y2],[x1,y1]]

class Circle():
    def __init__(self,c=[0,0],r=0):
        self.centre = c
        self.R = r

class Tire(Circle):
    def __init__(self,c=[0,0],rI=0,rO=0):
        Circle.__init__(self,c)
        self.rInner = rI
        self.rOuter = rO

class Fork(Tube):
    def __init__(self):
        self.type = 'suspension'
        self.axle2crown = 0
        self.offset = 0

    def calcOffset(self,headtube):
        # convention is pt1 is to the left of pt2
        offsetAngle = np.arcsin( (self.pt2[0]-self.pt1[0]) / self.axle2crown) + (headtube.A-np.pi/2)
        self.offset = self.axle2crown * np.sin(offsetAngle)
        print(('offset = %.1f cm' % PX2CM(self.offset)))

class Rider():
    def __init__(self,anthro):
        self.leg = anthro[0]
        self.torso = anthro[1]
        self.arm = anthro[2]

class Geometry():
    def __init__(self):
        self.T = Tubeset()
        self.fw = Tire()
        self.rw = Tire()
        self.cr = Circle()
        self.fork = Fork()
        self.paramlist = ['toptube','frontcentre','chainstay','rearcentre','reach','stack',
                        'bbdrop','headtube','htA','stA','standover','wheelbase','com','trail']
        self.params = dict(zip(self.paramlist,np.zeros(len(self.paramlist))))

    def addRider(self,anthro):
        self.rider = Rider(anthro)
        return

# for centre of mass calculation
# coords$stem <- c(coords$htXtt[1]+geo$stem/10 * cos(geo$stemA + pi/2-geo$HA),
#                    coords$htXtt[2]+geo$stem/10 * sin(geo$stemA + pi/2-geo$HA))
#   coords$saddle <- c(coords$BB[1]+cos(geo$SA)*geo$crank.length-cos(geo$SA)*anthro$leg*0.883,
#                      coords$BB[2]-sin(geo$SA)*geo$crank.length + sin(geo$SA)*anthro$leg*0.883)
#   saddle2bar <- sqrt((coords$stem[1]-coords$saddle[1])^2 + (coords$stem[2]-coords$saddle[2])^2)
#   # given seat and bar, find the angle of the torso. add torso curvature option
#   A_torso = acos((saddle2bar^2+anthro$torso^2-anthro$arm^2)/(2*saddle2bar*anthro$torso))
#   # shoulder at the apex of torso arm triangle
#   coords$shldr <- c(coords$saddle[1]+anthro$torso*cos(A_torso),coords$saddle[2]+anthro$torso*sin(A_torso)) 

    def calcParams(self):

        # this should already be a mean value
        R = np.mean([self.fw.rOuter,self.rw.rOuter])
        self.params['trail'] = PX2CM((R+self.fork.pt2[1]-self.T.tubes['ht'].pt2[1]) / np.sin(self.T.tubes['ht'].A) * np.cos(self.T.tubes['ht'].A) - (self.fork.pt2[0]-self.T.tubes['ht'].pt2[0]))
        self.params['reach'] = PX2CM(self.T.tubes['ht'].pt1[0] - self.T.tubes['st'].pt2[0])
        self.params['stack'] = PX2CM(self.T.tubes['st'].pt2[1] - self.T.tubes['ht'].pt1[1])
        self.params['bbdrop'] = PX2CM(self.T.tubes['st'].pt2[1] - self.fw.centre[1])
        self.params['htA'] = RAD2DEG(self.T.tubes['ht'].A)
        self.params['stA'] = RAD2DEG(self.T.tubes['st'].A)
        self.params['toptube'] = PX2CM(self.T.tubes['ht'].pt1[0] - self.T.tubes['st'].x(self.T.tubes['ht'].pt1[1]))
        self.params['headtube'] = PX2CM(self.T.tubes['ht'].l())
        self.params['frontcentre'] = PX2CM(self.fw.centre[0] - self.T.tubes['st'].pt2[0])
        self.params['rearcentre'] = PX2CM(self.T.tubes['st'].pt2[0] - self.rw.centre[0])
        self.params['chainstay'] = PX2CM(self.T.tubes['cs'].l())
        self.params['wheelbase'] = PX2CM(self.fw.centre[0] - self.rw.centre[0])
        # doesn't include top tube radius yet
        self.params['standover'] = PX2CM(np.mean([self.T.tubes['cs'].pt1[1],self.fork.pt2[1]]) - (self.T.tubes['tt'].m * self.T.tubes['st'].pt2[0] + self.T.tubes['tt'].b) + R)
        self.params['cob'] = self.params['frontcentre'] - self.params['wheelbase']/2
        # self.params['com'] =  self.calcCom()

    def printParams(self):
        for p1,p2 in zip(*[iter(self.paramlist)]*2):
            print('%15s  %8.1f %15s  %8.1f' % (p1,self.params[p1],p2,self.params[p2]))

    def plotTubes(self,aimg,tubeset):
        for tube in tubeset.tubes.keys():
            cv2.line(aimg,tuple(tubeset.tubes[tube].pt1.astype(int)),tuple(tubeset.tubes[tube].pt2.astype(int)),(0,255,0),CM2PX(0.6))
        cv2.line(aimg,tuple(self.fork.pt1.astype(int)),tuple(self.fork.pt2.astype(int)),(0,0,255),CM2PX(0.6))
        return(aimg)

    # def calcCom(self):
    #     com_leg = c((coords$BB[1]-coords$saddle[1])/2+coords$saddle[1],(coords$saddle[2]-coords$BB[2])/2+coords$BB[2])
    #     com_torso = c((coords$shldr[1]-coords$saddle[1]/2)+coords$saddle[1],(coords$shldr[2]-coords$saddle[2])/2+coords$saddle[2])
    #     com_arm = c((coords$stem[1]-coords$shldr[1])/2+coords$shldr[1],(coords$shldr[2]-coords$stem[2])/2+coords$stem[2])
    #     com <- c((com_leg[1]*anthro$leg + com_torso[1]*anthro$torso + com_arm[1]*anthro$arm)/sum(anthro),
    #             (com_leg[2]*anthro$leg + com_torso[2]*anthro$torso + com_arm[2]*anthro$arm)/sum(anthro))
   


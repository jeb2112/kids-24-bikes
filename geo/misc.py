from operator import eq
from ssl import PROTOCOL_TLSv1_1
import numpy as np
import re
import cv2
import matplotlib.pyplot as plt

figNo = 1
# rows = 0 
# cols = 0

class Convert():
    def __init__(self,mmpx=None):
        if mmpx is not None:
            self.mmpx = mmpx
    def CM2PX(self,cm):
        return(int(np.round(cm/(self.mmpx/10))))
    def PX2CM(self,px):
        return(px * self.mmpx/10.)
    def PX2MM(self,px):
        return(px * self.mmpx)

class Circle():
    def __init__(self,c=[0,0],r=0):
        self.centre = c
        self.R = r

class Tire(Circle):
    def __init__(self,c=[0,0],rI=0,rO=0):
        Circle.__init__(self,c)
        self.rInner = rI
        self.rOuter = rO

class Rider():
    def __init__(self,anthro):
        self.leg = anthro[0]
        self.torso = anthro[1]
        self.arm = anthro[2]


#################
# general methods
#################

def DEG2RAD(d):
    return(d * np.pi/180)
def RAD2DEG(r):
    return(r * 180./np.pi)

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

# for a text file list of images and mmpx scale factors
def readFile(filename):
    if filename.closed:
        f = open(filename,'r')
    else:
        f = filename
    flist=[]
    for line in f:
        flist.append([re.search('^[a-zA-Z0-9\/\-]*\.(png|jpg)',line).group(0),float(re.search('[0|1]\.[0-9]*',line).group(0))])
    f.close()
    return(flist)


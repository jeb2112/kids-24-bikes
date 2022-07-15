from operator import eq
from ssl import PROTOCOL_TLSv1_1
import numpy as np
import re
import cv2
import matplotlib.pyplot as plt

figNo = 1

class Cursor2:
    """
    A cross hair cursor.
    """
    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()

class Convert():
    def __init__(self,mmpx=None):
        if mmpx is not None:
            self.mmpx = mmpx # mm/pixel
        # scaling factors as products
        self.cm2px = 10 / self.mmpx
        self.px2cm = self.mmpx / 10
        self.px2mm = 1 / self.mmpx
    def CM2PX(self,cm):
        return int(np.round(cm * self.cm2px))
    def PX2CM(self,px):
        return px * self.px2cm
    def PX2MM(self,px):
        return px * self.mmpx
    def set_mmpx(self,mmpx):
        self.mmpx = mmpx

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
def cm2in(c):
    return c/2.54
def in2cm(i):
    return i*2.54

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


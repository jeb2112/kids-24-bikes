from operator import eq
from ssl import PROTOCOL_TLSv1_1
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import json,pickle
import os
import copy
# relative intra-package import. can use .misc for module in the same directory as this file,
# or geo.misc as well, resolves to the same thing.
from .misc import *
from geo.display import Display
from geo.tube import Fork,Tubeset
from geo.cvtubeset import CVTubeset


# base class for a complete geometry
class Geometry():
    def __init__(self,mmpx=None,name=None):
        self.T = None
        self.fw = None
        self.rw = None
        self.cr = None
        self.fork = Fork(mmpx=mmpx)
        self.paramlist = ['toptube','frontcentre','chainstay','rearcentre','reach','stack',
                        'bbdrop','headtube','htA','stA','standover','wheelbase','com','trail']
        self.params = dict(zip(self.paramlist,np.zeros(len(self.paramlist))))
        if mmpx is not None:
            self.cv = Convert(mmpx=mmpx)
            self.D = Display(mmpx=mmpx)
        self.name = name

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
        self.params['trail'] = self.cv.PX2CM((R+self.fork.pt2[1]-self.T.tubes['ht'].pt2[1]) / np.sin(self.T.tubes['ht'].A) * np.cos(self.T.tubes['ht'].A) - (self.fork.pt2[0]-self.T.tubes['ht'].pt2[0]))
        self.params['reach'] = self.cv.PX2CM(self.T.tubes['ht'].pt1[0] - self.T.tubes['st'].pt2[0])
        self.params['stack'] = self.cv.PX2CM(self.T.tubes['st'].pt2[1] - self.T.tubes['ht'].pt1[1])
        self.params['bbdrop'] = self.cv.PX2CM(self.T.tubes['st'].pt2[1] - self.fw.centre[1])
        self.params['htA'] = RAD2DEG(self.T.tubes['ht'].A)
        self.params['stA'] = RAD2DEG(self.T.tubes['st'].A)
        self.params['toptube'] = self.cv.PX2CM(self.T.tubes['ht'].pt1[0] - self.T.tubes['st'].x(self.T.tubes['ht'].pt1[1]))
        self.params['headtube'] = self.cv.PX2CM(self.T.tubes['ht'].l())
        self.params['frontcentre'] = self.cv.PX2CM(self.fw.centre[0] - self.T.tubes['st'].pt2[0])
        self.params['rearcentre'] = self.cv.PX2CM(self.T.tubes['st'].pt2[0] - self.rw.centre[0])
        self.params['chainstay'] = self.cv.PX2CM(self.T.tubes['cs'].l())
        self.params['wheelbase'] = self.cv.PX2CM(self.fw.centre[0] - self.rw.centre[0])
        # doesn't include top tube radius yet
        self.params['standover'] = self.cv.PX2CM(np.mean([self.T.tubes['cs'].pt1[1],self.fork.pt2[1]]) - (self.T.tubes['tt'].m * self.T.tubes['st'].pt2[0] + self.T.tubes['tt'].b) + R)
        self.params['cob'] = self.params['frontcentre'] - self.params['wheelbase']/2
        # self.params['com'] =  self.calcCom()

    def printParams(self):
        for p1,p2 in zip(*[iter(self.paramlist)]*2):
            print('%15s  %8.1f %15s  %8.1f' % (p1,self.params[p1],p2,self.params[p2]))

    def plotTubes(self,aimg,tubeset):
        for tube in tubeset.tubes.keys():
            cv2.line(aimg,tuple(tubeset.tubes[tube].pt1.astype(int)),tuple(tubeset.tubes[tube].pt2.astype(int)),(0,255,0),self.cv.CM2PX(0.6))
        cv2.line(aimg,tuple(self.fork.pt1.astype(int)),tuple(self.fork.pt2.astype(int)),(0,0,255),self.cv.CM2PX(0.6))
        return(aimg)

    def findwheels(self):
        pass

    def findlines(self):
        pass

    def save(self):
        fp = open(os.path.join('/home/src/kids-24-bikes/testdata/',self.name),'wb')
        pickle.dump(self,fp)

    # def calcCom(self):
    #     com_leg = c((coords$BB[1]-coords$saddle[1])/2+coords$saddle[1],(coords$saddle[2]-coords$BB[2])/2+coords$BB[2])
    #     com_torso = c((coords$shldr[1]-coords$saddle[1]/2)+coords$saddle[1],(coords$shldr[2]-coords$saddle[2])/2+coords$saddle[2])
    #     com_arm = c((coords$stem[1]-coords$shldr[1])/2+coords$shldr[1],(coords$shldr[2]-coords$stem[2])/2+coords$stem[2])
    #     com <- c((com_leg[1]*anthro$leg + com_torso[1]*anthro$torso + com_arm[1]*anthro$arm)/sum(anthro),
    #             (com_leg[2]*anthro$leg + com_torso[2]*anthro$torso + com_arm[2]*anthro$arm)/sum(anthro))
   

# sub-class for annotating a geometry
class AnnoGeometry(Geometry):
    def __init__(self,mmpx=None,name=None):
        super(AnnoGeometry,self).__init__(mmpx=mmpx,name=name)
        self.T = Tubeset()

    def findwheels(self,P):
        fig,ax = self.D.plotfig(P.imRGB,show=False)
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
        fig.set_size_inches(2*fig.get_size_inches())
        if False:
            cursor = Cursor2(ax)
            fig.canvas.mpl_connect('motion_notify_event',cursor.on_mouse_move)
        else:
            cursor = Cursor(ax)
        # enter 2 wheels 
        pts = plt.ginput(n=2,timeout=-1)
        plt.close()
        # sort x coord of the input points for left to right, back wheel first
        pts.sort(key=lambda pt:pt[0])
        # assume hubs at equal y-axis, average
        pts = [np.array([p[0],np.mean(pts,axis=0)[1]]) for p in pts]
        self.rw = Tire(pts[0],self.cv.CM2PX(in2cm(12)),self.cv.CM2PX(in2cm(13)))
        self.fw = Tire(pts[1],self.cv.CM2PX(in2cm(12)),self.cv.CM2PX(in2cm(13)))
        # calculate mmpx from image size and wheelbase spec.

    def findlines(self,P):
        fig,ax = self.D.plotfig(P.imRGB,show=False)
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
        fig.set_size_inches(2*fig.get_size_inches())
        if False:
            cursor = Cursor2(ax)
            fig.canvas.mpl_connect('motion_notify_event',cursor.on_mouse_move)
        else:
            cursor = Cursor(ax)
        # enter seat/upper, bottombracket, head/upper, head/lower
        pts = plt.ginput(n=4,timeout=-1)
        plt.close()
        pts.sort(key=lambda pt:pt[0])
        # convert tuples to arrays
        pts = [np.array(p) for p in pts]
        self.cr = Circle(pts[1],self.cv.CM2PX(in2cm(3)))
        self.T.tubes['tt'].setpts(pts[0],pts[2])
        self.T.tubes['ss'].setpts(self.rw.centre,pts[0])
        self.T.tubes['cs'].setpts(self.rw.centre,pts[1])
        self.T.tubes['st'].setpts(pts[0],pts[1])
        self.T.tubes['dt'].setpts(pts[1],pts[3])
        self.T.tubes['ht'].setpts(pts[2],pts[3])
        self.fork.setpts(pts[3],self.fw.centre)

    def calc(self):
        # create output
        self.calcParams()
        self.printParams()

    def plot(self,P):
        # final block with blocking
        P.imW = np.copy(P.imRGB)
        P.imw = self.plotTubes(P.imW,self.T)
        self.D.plotfig(P.imw,True)

        return P


# sub-class for cv processing a geometry
class CVGeometry(Geometry):
    def __init__(self,mmpx=None,name=None):
        super(CVGeometry,self).__init__(mmpx=mmpx,name=name)
        self.T = CVTubeset(mmpx=mmpx)

    def findwheels(self,P):
        wheels,chainring = P.houghCircles(P.imGRAY)
        # note sort. back wheel  returned first
        # decided not to use outer radius in hub estimation becuase tread pattern sometimes throws it off
        self.rw = Tire(np.mean(wheels[0:4:4],axis=0)[0:2],wheels[0,2],wheels[2,2])
        self.fw = Tire(np.mean(wheels[1:5:4],axis=0)[0:2],wheels[1,2],wheels[3,2])
        self.cr = Circle(chainring[0:2],chainring[2])

        # create working image with wheels masked out to unclutter for line detect
        P.imW = np.copy(P.imGRAY)
        P.imW = cv2.blur(P.imW,(5,5))
        # [0]have an extra dimension to get rid of here... 
        # P.maskCircle(np.concatenate((wheels,self.chainring),axis=0),P.imW)
        # P.maskCircle(np.reshape(np.append(self.rw.centre,self.rw.rOuter),(1,3)),P.imW)
        # P.maskCircle(np.reshape(np.append(self.cr.centre,self.cr.R),(1,3)),P.imW)
        # mantra. increase these radii
        P.maskCircle(np.reshape(np.append(self.fw.centre,self.rw.rOuter*1.1),(1,3)),P.imW)
        P.maskCircle(np.reshape(np.append(self.rw.centre,self.rw.rOuter*1.1),(1,3)),P.imW)
        P.maskCircle(np.reshape(np.append(self.cr.centre,self.cr.R*1.1),(1,3)),P.imW)
        # try to save more of the seatpost here 
        P.maskRect([(self.rw.centre[0],0,self.cr.centre[0],self.rw.centre[1]-self.rw.rOuter)],P.imW)
        return P

    def findlines(self,P):
        # try preliminary thresh to eliminate paint graphic effects
        ret,P.imW = cv2.threshold(P.imW,240,255,cv2.THRESH_BINARY)
        # start with main tubes. old method.
        # avglines,meqs = P.houghLines(P.imW,P.imRGB,minlength=8.5)
        # frog62, cube240. increase min length, one way to avoid the chain confusing the top tube
        # should generally be a correct strategy although it fails for riprock with such a bent top tube.
        # another round of refinement on angles better. or should just improve the maskCircle. 
        avglines,meqs = P.houghLines(P.imW,P.imRGB,minlength=11.0)    
        self.T.assignTubeLines(avglines,meqs,['tt','dt'])

        # todo: create ROI with seat tube only
        # find seat tube. new method
        lines = P.houghLinesS(P.imW,minlength=10)
        P.imW = np.copy(P.imRGB)
        self.D.plotLines(P.imW,lines,False,title="seat tube line detection")
        self.T.createSeatTubeTarget(self.cr)
        self.T.assignSeatTubeLine(lines)

        # todo:  head tube ROI. mask out everything to the left of the tt/dt intersection
        # find head tube
        # metaHT. increase minlength 4 to 7, separate edge process (maybe not needed though)
        # try preliminary thresh to eliminate paint graphic effects
        # pineridge. increase threshold due to white lettering
        ret,P.imW = cv2.threshold(P.imW,245,255,cv2.THRESH_BINARY)
        lines = P.houghLinesS(P.imW,minlength=7,edgeprocess='head')
        P.imW = np.copy(P.imRGB)
        self.T.createHeadTubeTarget(self.fw,type='susp')
        self.T.assignHeadTubeLine(lines)
        # np.concatenate((lines,np.reshape(self.T.tubes['ht'].pt1+self.T.tubes['ht'].pt2,(1,1,4))),axis=0)
        self.D.plotLines(P.imW,lines,False,title="head tube line detection")
        self.D.plotLines(P.imW,np.reshape(self.T.tubes['ht'].pt1+self.T.tubes['ht'].pt2,(1,1,4)).astype(int),False,title="head tube line detection",color=(255,0,0))

        P.imW=np.copy(P.imRGB)
        self.T.plotTubes(P.imW)
        plt.show(block= not __debug__)

        # set the tubes lengths according to intersections
        self.T.calcTubes()

        # process the fork for refinement of head tube angle. 
        P.imW = np.copy(P.imGRAY)
        lines,meqs = P.findForkLines(self.fw,minlength=5)
        # self.T.modifyTubeLines(meqs,'ht',op='mean')
        # alite. Creig-24. charger. mxxc. head tube estimate not good enough use fork only
        self.T.modifyTubeLines(meqs,'ht',op='replace')

        # further adjust the current head tube estimate using the right edge of the head tube
        # this could be done with thresholded silhouette
        # this might replace the fork lines estimate?
        P.imW=np.copy(P.imGRAY)
        ret,P.imW = cv2.threshold(P.imW,245,255,cv2.THRESH_BINARY)
        self.T.shiftHeadTube(P.imW)

        # recalc after modfication 
        self.T.calcTubes()

        P.imW=np.copy(P.imRGB)
        self.T.plotTubes(P.imW)
        plt.show(block= not __debug__)

        # find the length of the head tube 
        P.imW = np.copy(P.imRGB)
        # Creig-24. slight error because brake cable is below the bottom of tube. need an extra check on tube profile perpendicular
        self.T.extendHeadTube(P.imW)
        # add fork, chainstay, seatstay
        self.T.addStays(self.rw.centre)
        self.fork.pt1 = self.T.tubes['ht'].pt2
        self.fork.pt2 = self.fw.centre

        P.imW=np.copy(P.imRGB)
        P.imW = self.plotTubes(P.imW,self.T)
        self.D.plotfig(P.imW,False,title="head extend")

        # with head tube approximately correct, redo the head angle estimate with better measurement.
        # P.imW = np.copy(P.imRGB)
        # edge. color should be less needed for this profile defined by white background
        P.imW = np.copy(P.imGRAY)
        # mxtrail. try thresholding for this measurement. 
        # vertex. a small white reflection of black paint requires this threshold
        ret,P.imW = cv2.threshold(P.imW,240,255,cv2.THRESH_BINARY)
        meq = self.T.measureHeadTube(P.imW)
        if meq is not None:       
            self.T.modifyTubeLines(meq,'ht',op='mean')
            # kato. poor initial and very good secondary detection. a better combination might be averaging the slopes
            # whille allowing the centreline to be entirely goverened by the secondary detection
            # self.T.modifyTubeLines(meq,'ht',op='replace')

        # replot the tubelines
        P.imW=np.copy(P.imRGB)
        self.T.plotTubes(P.imW)

        # recalc the tubes
        self.T.calcTubes()
        P.imW = np.copy(P.imRGB)
        # reuse of extendHeadTube ought to work but might hit welding bumps
        self.T.extendHeadTube(P.imW)
        
        # redo fork
        self.fork.pt1 = self.T.tubes['ht'].pt2
        self.fork.axle2crown = self.fork.l()
        self.fork.calcOffset(self.T.tubes['ht'])

        # create output
        self.calcParams()
        self.printParams()

        # final block with blocking
        P.imW = np.copy(P.imRGB)
        P.imw = self.plotTubes(P.imW,self.T)
        self.D.plotfig(P.imw,True)

        return P
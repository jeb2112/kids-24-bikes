import os
import sys
import io
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
import peakutils
import scipy.interpolate

figNo = 1

class profilePhoto():
    def __init__(self,filename):
        self.rows = 0
        self.cols = 0
        self.filename = filename
        self.imRGB = None
        self.imGRAY = None
        self.imEDGE = None
        self.imANNO = None
        self.imW = None
        self.loadImage(self.filename)

    def loadImage(self,filename):
        im = imageio.imread(filename)
        # if png/alpha, remove alpha
        self.rows,self.cols = np.shape(im)[0:2]
        if np.shape(im)[2]==4:
            im = cv2.cvtColor(im,cv2.COLOR_BGRA2BGR)
        self.imRGB = im 
        self.imANNO = self.imRGB
        self.imGRAY = cv2.cvtColor(self.imRGB,cv2.COLOR_BGR2GRAY)

    def houghCircles(self,bw):
        # this preblur helps get rid of the apparent line in the middle of a tube
        # due to the reflection of light
        bw = cv2.blur(bw,(5,5))
        # not sure about final param1,2 choices yet
        wheelsInner = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,
            1,minDist=200,param1=70,param2=50,minRadius=40,maxRadius=120)
        wheelsOuter = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,
            1,minDist=200,param1=50,param2=40,minRadius=80,maxRadius=200)
        wheels = np.concatenate((wheelsInner,wheelsOuter),axis=1)
        chainring = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,
            1,minDist=200,param1=70,param2=50,minRadius=10,maxRadius=30)
        return wheels,chainring

    def maskCircle(self,masks,target):
        # masks. list of circles defined by centre point, radius
        # target. target image
        # have an extra 5 pixels hard-coded in here, but the fits returned by hough though are inordinately sensitive 
        # to this kludged number so further test needed.
        for c in masks:
            cv2.circle(target,(c[0],c[1]),int(c[2]+5.),255,-1)        
        
    def resetAnnotatedIm(self):
        self.imANNO = self.imRGB

    def houghLines(self,bw,aw,houghProcess='p'):
        global figNo
        # bw = cv2.blur(bw,(5,5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        for i in range(0,2):
            bw = cv2.dilate(bw,kernel,iterations=3)
            bw = cv2.erode(bw,kernel,iterations=2)

        edges = cv2.Canny(bw,100,200,apertureSize=3,L2gradient=True)
        plotFig(edges)

        # line processing
        if houghProcess=='p':
        # probabilistic works better for short segments like head tube. these numbers are all fine tuned
        # and hard-coded for cleary though.
        # 1. estimate of headset might be better by clipping the rigid fork exacttly at the headtube. or for suspension,
        # using the fork too.
        # 2. average line for the downtube in biased for the cleary example, can't see why the averaing doesn't worok
            lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=50)
            if (lines is not None):
                # plot raw lines detection
                bw2 = np.copy(aw)
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(bw2,(x1,y1),(x2,y2),(0,0,255),2)
                plotFig(bw2)

            # average matching pairs of lines. slope/intercept not working very well to select the matching pairs
            # becuase it is non-linear.
            # try rho/theta instead
            eqns = np.zeros((np.shape(lines)[0],2))
            rhotheta = np.zeros((np.shape(lines)[0],2))
            for i,line in enumerate(lines):
                eqns[i,:] = pts2eq(((line[0,0:2]),(line[0,2:4])))
                rhotheta[i,:] = coord2angle(line)

            meqs=None
            avglines=None
            while len(eqns)>0:
                # find all equations matching the slope of current first eqn in list. Using 15% as the matching threshold
                eqnset1 = np.where(np.abs(eqns[:,0]-eqns[0,0])<np.abs(.15*eqns[0,0]))[0][0:]
                # equal slope, equal offset. Using 2% to qualify as equal.
                eqnset1a =  eqnset1[np.where((np.abs(eqns[eqnset1,1]-eqns[0,1])<np.abs(0.01*eqns[0,1])))]
                # equal slope, different offset but close enough to be a matchingline
                # y intercept 10% as the threshold%. 10% too low. try 20%.
                # if change to rhotheta then can identify by a length in actual pixels
                eqnset1b = np.setdiff1d(eqnset1,eqnset1a)
                eqnset1b = eqnset1b[np.where(np.abs(eqns[eqnset1b,1]-eqns[0,1])<np.abs(0.2*eqns[0,1]))]
                # equal slope, different offset
                meq1 = np.mean(eqns[eqnset1a],axis=0)
                if len(eqnset1b) > 0:
                    meq2 = np.mean(eqns[eqnset1b],axis=0)
                    meq = np.mean([meq1,meq2],axis=0)
                else:
                    meq = meq1
                if meqs is None:
                    meqs = meq
                    avglines = eq2pts(meq,(0,600))
                else:
                    meqs = np.append(meqs,meq,axis=0)
                    avglines = np.append(avglines,eq2pts(meq,(200,400)),axis=0)
                # throw out only the used offsets 
                eqns = np.delete(eqns,np.concatenate((eqnset1a,eqnset1b),axis=0),axis=0)
                rhotheta = np.delete(rhotheta,np.concatenate((eqnset1a,eqnset1b),axis=0),axis=0)

            avglines = np.reshape(avglines,(len(meqs)/2,4))
            meqs = np.reshape(meqs,(len(meqs)/2,2))

        
            return(avglines,meqs)

        else:

            # HoughLines option hasn't been used recently, probably wont' need it...
            # target set of angles: head tube, seat tube, down tube, top tube  
            # conventional bike angles rotating from neg x to pos y are: (69,72,-47,-23)
            # normal angles measured from pos x rotating to pos y, where pos y is down in cv2 image: theta+90
            targetAngles = (159,162,43,67)
            # hardcoded for cleary.  these need to scale for image resolution
            targetRhos = (-300,-200,400,300)
            lines = cv2.HoughLines(edges,rho=1.0,theta=np.pi/180,threshold=60,srn=10,stn=10)
            if (lines is not None):
                for line in lines:
                    for rho,theta in line:
                        linexy = np.array(angle2coord(line)).astype(int)
                        cv2.line(aimg,tuple(linexy[0]),tuple(linexy[1]),(0,0,255),2)
                    plt.figure(figNo)
                    plt.imshow(aimg)
                    plt.show()
                    figNo += 1
                
            angles = lines[:,0,1]
            rhos = lines[:,0,0]
            # fit the angles histogram
            h,b = np.histogram(angles,bins=90,range=(0,np.pi))
            bx = b[:-1] + (b[1]-b[0])/2
            bxint = np.arange(bx[0],bx[-1],(bx[-1]-bx[0])/1000.0)
            bsp = scipy.interpolate.splrep(bx,h,w=np.ones(len(bx)),k=3,s=10.0)
            bspl = scipy.interpolate.splev(bxint,bsp)
            peakIndexes = peakutils.indexes(bspl,thres=0.02/max(bspl))
            peaks = np.sort(bspl[peakIndexes])[::-1][0:5]
            peakIndexes2 = peakIndexes[np.argsort(bspl[peakIndexes])[::-1][0:5]]
            estAngles = bxint[peakIndexes2]
            # convert normal angle for matching
            peakdegrees = estAngles * 180.0/np.pi
            print(peakdegrees)
            dtA = peakdegrees[np.abs(peakdegrees-targetAngles[2]).argmin()]
            ttA = peakdegrees[np.abs(peakdegrees-targetAngles[3]).argmin()]
            stA = peakdegrees[np.abs(peakdegrees-targetAngles[1]).argmin()]
            htA = peakdegrees[np.abs(peakdegrees-targetAngles[0]).argmin()]
            # convert back to radians 
            rA = (np.array((htA,stA,dtA,ttA)) ) * np.pi/180
            plt.figure(figNo),plt.subplot(2,1,1)
            figNo = figNo + 1
            plt.hist(angles,bins=90)
            plt.plot(bxint,bspl)
            plt.plot(bxint[peakIndexes],bspl[peakIndexes],'r+')
            plt.show(block=False)

            # fit the rhos histogram
            h,b = np.histogram(rhos,bins=90,range=(-np.shape(aimg)[0],np.shape(aimg)[0]))
            bx = b[:-1] + (b[1]-b[0])/2
            bxint = np.arange(bx[0],bx[-1],(bx[-1]-bx[0])/1000.0)
            bsp = scipy.interpolate.splrep(bx,h,w=np.ones(len(bx)),k=3,s=10.0)
            bspl = scipy.interpolate.splev(bxint,bsp)
            peakIndexes = peakutils.indexes(bspl,thres=0.02/max(bspl))
            peaks = np.sort(bspl[peakIndexes])[::-1][0:5] # top five only
            peakIndexes2 = peakIndexes[np.argsort(bspl[peakIndexes])[::-1][0:5]]
            estRhos = bxint[peakIndexes2]
            print(estRhos)
            dtR = estRhos[np.abs(estRhos-targetRhos[2]).argmin()]
            ttR = estRhos[np.abs(estRhos-targetRhos[3]).argmin()]
            stR = estRhos[np.abs(estRhos-targetRhos[1]).argmin()]
            htR = estRhos[np.abs(estRhos-targetRhos[0]).argmin()]
            rR = (htR,stR,dtR,ttR)

            plt.subplot(2,1,2)
            plt.hist(rhos,bins=90)
            plt.plot(bxint,bspl)
            plt.show(block=False)
            print(rR)

            # need to convert to return a linexy here
            return(bimg,aimg,rA,rR)


class Tube():
    def __init__(self):
        # pt1 is leftmost point, pt2 rightmost
        self.pt1 = np.array([0,0])
        self.pt2 = np.array([0,0])
        self.m = 0
        self.b = 0
        self.A = 0
        self.rho = 0
        self.theta = 0

class Tire():
    def __init__(self):
        self.centre = [0,0]
        self.rInner = 0
        self.rOuter = 0

class Ring():
    def __init__(self):
        self.centre = [0,0]

class Fork():
    def __init__(self):
        self.type = 'suspension'
        self.axle2crown = 0
        self.offset = 0
        self.pt1 = [0,0]
        self.pt2 = [0,0]

    def calcOffset(self,t):
        # need to ensure all tubes are defined left/right. or the tubeset class has the hub points as well
        m,b = pts2eq([t.tubes['ht'].pt1,t.tubes['ht'].pt2])
        htA = np.arctan(m) 
        offsetAngle = np.arcsin( (self.pt2[0]-self.pt1[0]) / self.axle2crown) + (htA-np.pi/2)
        self.offset = self.axle2crown * np.sin(offsetAngle)

class Tubeset():
    def __init__(self):
        self.tubes = dict(dt=Tube(),tt=Tube(),st=Tube(),ht=Tube(),cs=Tube(),ss=Tube())
        # target set of angles: head tube, seat tube, down tube, top tube  
        # conventional bike angles rotating from neg x to pos y are: (69,72,-47,-23)
        self.targetAngles = np.array([68,73,-47,-23])

    def assignTubeLines(self,avglines,meqs):
        # corresponding target slopes of line segments:
        targetSlopes = np.tan(self.targetAngles * np.pi/180)

        for i,tube in enumerate(['ht','st','dt','tt']):
            self.tubes[tube].pt1 = avglines[np.abs(meqs[:,0] - targetSlopes[i]).argmin()][0:2]
            self.tubes[tube].pt2 = avglines[np.abs(meqs[:,0] - targetSlopes[i]).argmin()][2:4]

    # redo function with just line segments from HoughP instead of normal rho,thetas from Hough
    # messy float/int/array/tuple problem. cv2 functions work with tuples of ints, but coordinate
    # calculations have to be arrays of floats. because only arrays can be cast back to ints for cv2.
    # coordinates of each tube in x1,y1,x2,y2
    # origin is cv2 top left
    def calcTubes(self):
        # seat tube intersecting top tube
        eq1 = pts2eq((self.tubes['st'].pt1,self.tubes['st'].pt2))
        eq2 = pts2eq((self.tubes['tt'].pt1,self.tubes['tt'].pt2))
        xint = (eq1[1]-eq2[1]) / (eq2[0] - eq1[0])
        self.tubes['st'].pt1 = np.array([xint,eq1[0]*xint+eq1[1]])
        self.tubes['tt'].pt1 = np.array([xint,eq1[0]*xint+eq1[1]])
        # toptube intersecting head tube
        eq1 = pts2eq((self.tubes['tt'].pt1,self.tubes['tt'].pt2))
        eq2 = pts2eq((self.tubes['ht'].pt1,self.tubes['ht'].pt2))
        xint = (eq1[1]-eq2[1]) / (eq2[0] - eq1[0])
        self.tubes['ht'].pt1 = np.array([xint,eq1[0]*xint+eq1[1]])
        self.tubes['tt'].pt2 = np.array([xint,eq1[0]*xint+eq1[1]])
        # headtube intersecting down tube
        eq1 = pts2eq((self.tubes['dt'].pt1,self.tubes['dt'].pt2))
        eq2 = pts2eq((self.tubes['ht'].pt1,self.tubes['ht'].pt2))
        xint = (eq1[1]-eq2[1]) / (eq2[0] - eq1[0])
        self.tubes['ht'].pt2 = np.array([xint,eq1[0]*xint+eq1[1]])
        self.tubes['dt'].pt2 = np.array([xint,eq1[0]*xint+eq1[1]])
        # seattube intersecting down tube
        eq1 = pts2eq((self.tubes['dt'].pt1,self.tubes['dt'].pt2))
        eq2 = pts2eq((self.tubes['st'].pt1,self.tubes['st'].pt2))
        xint = (eq1[1]-eq2[1]) / (eq2[0] - eq1[0])
        self.tubes['st'].pt2 = np.array([xint,eq1[0]*xint+eq1[1]])
        self.tubes['dt'].pt1 = np.array([xint,eq1[0]*xint+eq1[1]])

    def extendHeadTube(self,img):
        global figNo
        rows,cols = np.shape(img)[0:2]
        m,b = pts2eq([self.tubes['ht'].pt1,self.tubes['ht'].pt2])
        htA = np.arctan(m) * 180/np.pi
        # point of rotation 
        htx,hty = np.round(np.mean([self.tubes['ht'].pt1,self.tubes['ht'].pt2],axis=0)).astype(int)
        M = cv2.getRotationMatrix2D((htx,hty),htA-90,1)
        rimg = cv2.warpAffine(img,M,(cols,rows))
        # plotFig(rimg,True)
        # extent of profile again hard-coded for cleary. but it should calculate to +/- 6cm or so
        lrange = range(hty-25,hty+25)
        htprofile = rimg[lrange,htx]
        # fit spline to colour profiles
        bxint = np.round(np.arange(lrange[0],lrange[-1],.1)*10)/10
        bspl = np.zeros((len(bxint),3))
        ypeaks=[]
        for i in range(0,3):
            # arbitrary smoothing factor
            bsp = scipy.interpolate.splrep(lrange,htprofile[:,i],np.ones(len(lrange)),k=3,s=len(bxint))
            bspl[:,i] = scipy.interpolate.splev(bxint,bsp,der=1)
            ypeaks.append(bxint[list(peakutils.indexes(bspl[:,i],thres=0.5,min_dist=100))])

        # create detection search ranges. 3cm above/below the headtube/topdowntube intersection points.
        toprange = range(int(self.tubes['ht'].pt1[1])-7,int(self.tubes['ht'].pt1[1])-20,-1)
        # map search range from pixel units to the interpolated 0.1 pixel scale
        toprangeint = range(np.where(bxint==toprange[0])[0][0],np.where(bxint==toprange[-1])[0][0],-1)
        botrange = range(int(self.tubes['ht'].pt2[1])+7,int(self.tubes['ht'].pt2[1]+20))
        botrangeint = range(np.where(bxint==botrange[0])[0][0],np.where(bxint==botrange[-1])[0][0])
        # average the three colour bands
        mbspl = np.mean(np.abs(bspl),axis=1)

        # 0th index take the first peak in the derivative. note these thresholds are hard-coded
        # and will be too high for any bikes that are black
        toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.4,min_dist=10)[0]
        botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.4,min_dist=10)[0]+botrangeint[0] 
        plt.figure(figNo)
        figNo+=1
        plt.subplot(2,1,1)
        plt.plot(lrange,htprofile)
        plt.subplot(2,1,2)
        plt.plot(bxint,np.mean(np.abs(bspl),axis=1))
        plt.plot(bxint[botrangeint],mbspl[botrangeint],'r')
        plt.plot(bxint[toprangeint],mbspl[toprangeint],'r')
        plt.show(block=False)
    
        headtubetoplength = self.tubes['ht'].pt1[1] - bxint[toppeak]
        headtubebotlength = bxint[botpeak] - self.tubes['ht'].pt2[1]
        self.tubes['ht'].pt2 += np.array([np.cos(htA*np.pi/180)*headtubebotlength,np.sin(htA*np.pi/180)*headtubebotlength])
        self.tubes['ht'].pt1 -= np.array([np.cos(htA*np.pi/180)*headtubetoplength,np.sin(htA*np.pi/180)*headtubetoplength])

    def addStays(self,rearhub):
        # need to check order of wheels
        self.tubes['cs'].pt1 = np.array(rearhub)
        self.tubes['cs'].pt2 = self.tubes['st'].pt2
        self.tubes['ss'].pt1 = self.tubes['cs'].pt1
        self.tubes['ss'].pt2 = self.tubes['st'].pt1

    def plotTubes(self,aimg):
        # plot raw lines detection
        # bw2 = np.copy(bw)
        for tube in ['ht','st','dt','tt']:
            # for some reason can't follow this syntax with the line array as did with tuples
            #for x1,y1,x2,y2 in np.nditer(Lline):
            #    cv2.line(aimg2,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.line(aimg,tuple(self.tubes[tube].pt1.astype(int)),tuple(self.tubes[tube].pt2.astype(int)),(255,0,0),2)
        plotFig(aimg)

class Rider():
    def __init__(self,anthro):
        self.leg = anthro[0]
        self.torso = anthro[1]
        self.arm = anthro[2]

class Geometry():
    def __init__(self):
        self.tubes = Tubeset()
        self.fw = Tire()
        self.rw = Tire()
        self.cr = Ring()
        self.fork = Fork()
        self.trail = 0
        self.standover = 0
        self.com = 0
        self.cob = 0

    def addRider(self,anthro):
        self.rider = Rider(anthro)
        return

    def calcTrail(self,t,w):
        m,b = pts2eq((t.tubes['ht'].pt1,t.tubes['ht'].pt2))
        htA = np.arctan(m) 
        wheelRadius = np.mean(w[0,2:4,2])
        # top of head tube is currently point 1 instead of 0
        self.trail = (wheelRadius+self.fork.pt2[1]-t.tubes['ht'].pt2[1]) / np.sin(htA) * np.cos(htA) - (self.fork.pt2[0]-t.tubes['ht'].pt2[0])

    def calcStandover(self,t,w):
        wheelRadius = np.mean(w[0,2:4,2])
        m,b = pts2eq((t.tubes['tt'].pt1,t.tubes['tt'].pt2))
        self.standover = t.tubes['st'].pt2[1] - (m * t.tubes['st'].pt2[0] + b) + wheelRadius

    def plotTubes(self,aimg,tubeset):
        for tube in tubeset.tubes.keys():
            cv2.line(aimg,tuple(tubeset.tubes[tube].pt1.astype(int)),tuple(tubeset.tubes[tube].pt2.astype(int)),(0,255,0),2)
        cv2.line(aimg,tuple(self.fork.pt1.astype(int)),tuple(self.fork.pt2.astype(int)),(0,0,255),2)
        return(aimg)
   

def pts2eq(((x1,y1),(x2,y2))):
    m = float(y2-y1)/float(x2-x1)
    b = y1 - m * x1
    return [m,b]

def eq2pts((m,b),(x1,x2)):
    y1 = m*x1 + b
    y2 = m*x2 + b
    return ([x1,y1],[x2,y2])

def plotFig(img,blockFlag=False):
    global figNo
    plt.figure(figNo)
    plt.imshow(img)
    plt.show(block=blockFlag)
    figNo += 1

def coord2angle(line):
    for x1,y1,x2,y2 in line:
        m = float(y2-y1) / float(x2-x1)
        b = y1 - m*x1
        theta = np.pi/2 - np.arctan(m)
        rho = -b / (m*np.cos(theta) - np.sin(theta)) 
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
    return[[x1,y1],[x2,y2]]


if __name__=='__main__':
    filename = sys.argv[1]
    P = profilePhoto(filename)
    G = Geometry()
    # aimg2 = np.copy(aimg)
    wheels,G.chainring = P.houghCircles(P.imGRAY)
    G.fw.inner = wheels[0][0]
    G.rw.inner = wheels[0][1]
    G.fw.outer = wheels[0][2]
    G.rw.outer = wheels[0][3]
    # create working image with wheels masked out to unclutter for line detect
    P.imW = np.copy(P.imGRAY)
    P.imW = cv2.blur(P.imW,(5,5))
    # [0]have an extra dimension to get rid of here... 
    P.maskCircle(np.concatenate((wheels,G.chainring),axis=1)[0],P.imW)

    # built cv2 for imshow GTK UI support
    # but waitKey and destroyAllWindows are clunky use matplotlib for now
    # cv2.startWindowThread()
    # cv2.namedWindow('Circles Image')
    # cv2.imshow('Circles Image',houghCircleImage)
    # cv2.waitKey(1)
    #cv2.destroyAllWindows()
    # modified this to return line coords instead of rho/theta normals
    avglines,meqs = P.houghLines(P.imW,P.imRGB)
    tubeset = Tubeset()
    tubeset.assignTubeLines(avglines,meqs)
    tubeset.calcTubes()

    # improve head tube detection
    P.imW = np.copy(P.imRGB)
    tubeset.extendHeadTube(P.imW)
    tubeset.addStays(wheels[0][1][0:2])
    # add fork, chainstay, seatstay
    G.fork.pt1 = tubeset.tubes['ht'].pt2
    G.fork.pt2 = wheels[0][0][0:2]
    G.fork.axle2crown = np.sqrt(pow(G.fork.pt1[0]-G.fork.pt2[0],2.0)+pow(G.fork.pt1[1]-G.fork.pt2[1],2.0))
    
    P.imW = np.copy(P.imRGB)
    P.imw = G.plotTubes(P.imW,tubeset)
    plotFig(P.imw,True)

    G.fork.calcOffset(tubeset)
    print('offset =',G.fork.offset)
    G.calcTrail(tubeset,wheels)
    print('trail = ',G.trail)
    G.calcStandover(tubeset,wheels)
    print('standover = ',G.standover)

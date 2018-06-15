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

    def maskCircle(self,masks,target):
        # masks. list of circles defined by centre point, radius
        # target. target image
        # have an extra 3 pixels hard-coded in here
        for c in masks:
            cv2.circle(target,(c[0],c[1]),int(c[2]+3.),255,-1)        
        
    def resetAnnotatedIm(self):
        self.imANNO = self.imRGB

class Tube():
    def __init__(self):
        # pt1 is leftmost point, pt2 rightmost
        self.pt1 = [0,0]
        self.pt2 = [0,0]
        self.m = 0
        self.b = 0
        self.A = 0

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

class Tubeset():
    def __init__(self):
        self.dt = Tube()
        self.tt = Tube()
        self.st = Tube()
        self.ht = Tube()
        self.cs = Tube()
        self.ss = Tube()

class Rider():
    def __init__(self,anthro):
        self.leg = anthro[0]
        self.torso = anthro[1]
        self.arm = anthro[2]

class Geometry():
    def __init__(self):
        self.tubes = Tubeset()
        self.fw = Wheel()
        self.rw = Wheel()
        self.cr = Ring()
        self.fork = Fork()
        self.trail = 0
        self.standover = 0
        self.rider = Rider()
        self.com = 0
        self.cob = 0


def plotTubes(aimg,tubeset):
    for pt1,pt2 in tubeset:
        cv2.line(aimg,tuple(pt1.astype(int)),tuple(pt2.astype(int)),(0,255,0),2)
    return(aimg)

# redo function with just line segments from HoughP instead of normal rho,thetas from Hough
# messy float/int/array/tuple problem. cv2 functions work with tuples of ints, but coordinate
# calculations have to be arrays of floats. because only arrays can be cast back to ints for cv2.
# coordinates of each tube in x1,y1,x2,y2
# origin is cv2 top left
def calcTubes2(tlines,wheels,chainrings):
    frontwheel = np.array([wheels[0,0,0],wheels[0,1,1]])
    # need to check order of wheels
    backwheel = wheels[0,1,0:2]
    bottombracket = chainring[0,0,0:2]
    chainstay = np.array([[backwheel[0],backwheel[1]],[bottombracket[0],bottombracket[1]]])
    # seat tube intersecting top tube
    eq1 = pts2eq([tlines[3][0:2],tlines[3][2:4]])
    eq2 = pts2eq([tlines[1][0:2],tlines[1][2:4]])
    xint = (eq1[1]-eq2[1]) / (eq2[0] - eq1[0])
    seattube = np.array([[bottombracket[0],bottombracket[1]],[xint,eq1[0]*xint+eq1[1]]])
    # toptube intersecting head tube
    eq2 = pts2eq([tlines[0][0:2],tlines[0][2:4]])
    xint = (eq1[1]-eq2[1]) / (eq2[0] - eq1[0])
    toptube = np.array([seattube[1],[xint,eq1[0]*xint+eq1[1]]])
    # downtube intersecting head tube
    eq1 = pts2eq([tlines[2][0:2],tlines[2][2:4]])
    xint = (eq1[1]-eq2[1]) / (eq2[0] - eq1[0])
    downtube = np.array([seattube[0],[xint,eq1[0]*xint+eq1[1]]])
    # true length of headtube should be estimated from a line profile along the known axis
    # picking up headset races or paint colour boundary
    headtube = np.array([downtube[1],toptube[1]])
    return [seattube,headtube,downtube,toptube]

# original version with normals
def calcTubes(tangles,trhos,wheels,chainring):
    # convert normal angles back to bike convention:
    bangles = -(np.pi/2 - tangles)
    frontwheel = np.array([wheels[0,0,0],wheels[0,1,1]])
    # need to check order of wheels
    backwheel = wheels[0,1,0:2]
    bottombracket = chainring[0,0,0:2]
    chainstay = np.array([[backwheel[0],backwheel[1]],[bottombracket[0],bottombracket[1]]])
    toptube = np.array(angle2coord(np.reshape(np.array((trhos[3],tangles[3])),(1,2))))
    lseattube = 130 # arbitrary number
    # by using conventional bike angles, have to add negative signs to maintain consistency. should complement the
    # bike angles to take care of this.
    seattube = np.array([[bottombracket[0],bottombracket[1]],[bottombracket[0]+(-1)*lseattube*np.cos(bangles[1]),bottombracket[1] + 
        (-1)*lseattube*np.sin(bangles[1])]])
    # seat tube intersecting toptube
    eq1 = pts2eq(toptube)
    print('toptube eqn',eq1)
    eq2 = pts2eq(seattube)
    xint = (eq1[1]-eq2[1]) / (eq2[0] - eq1[0])
    seattube = np.array([[bottombracket[0],bottombracket[1]],[xint,eq1[0]*xint+eq1[1]]])
    toptube[0] = seattube[1]
    # toptube intersecting forkheadtube
    lforkhead = 330
    forkhead = np.array([[frontwheel[0],frontwheel[1]],[frontwheel[0]+(-1)*lforkhead*np.cos(bangles[0]),
        frontwheel[1]+(-1)*lforkhead*np.sin(bangles[0])]])
    eq2 = pts2eq(forkhead)
    xint = (eq1[1]-eq2[1]) / (eq2[0] - eq1[0])
    forkhead = np.array([[frontwheel[0],frontwheel[1]],[xint,eq1[0]*xint+eq1[1]]])
    toptube[1] = forkhead[1]
    return [chainstay,toptube,seattube,forkhead]
    

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

def houghLines(bw,houghProcess='p'):
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
    # probabilistic works better for short segeements like head tube. these numbers are all fine tuned
    # and hard-coded for cleary though.
        lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=50)
        if (lines is not None):
            # plot raw lines detection
            bw2 = np.copy(bw)
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(bw2,(x1,y1),(x2,y2),(0,0,255),2)
            plotFig(bw2)

        # average matching pairs of lines. this is not working very well for headtube, 
        eqns = np.zeros((np.shape(lines)[0],2))
        for i,line in enumerate(lines):
            eqns[i,:] = pts2eq(((line[0,0:2]),(line[0,2:4])))

        meqs=None
        avglines=None
        while len(eqns)>0:
            # find all equations matching the current first eqn in list. Using 5% as the matching threshold
            eqnset1 = np.where(np.abs(eqns[:,0]-eqns[0,0])<np.abs(.05*eqns[0,0]))[0][0:]
            # equal slope, equal offset. Using 1% to qualify as equal
            eqnset1a =  eqnset1[np.where((np.abs(eqns[eqnset1,1]-eqns[0,1])<np.abs(0.01*eqns[0,1])))]
            # equal slope, different offset but close enough to be a matchingline
            eqnset1b = np.setdiff1d(eqnset1,eqnset1a)
            # y intercept 10% as the threshold%
            eqnset1b = eqnset1b[np.where(np.abs(eqns[eqnset1b,1]-eqns[0,1])<np.abs(0.1*eqns[0,1]))]
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
                avglines = np.append(avglines,eq2pts(meq,(0,600)),axis=0)
            # keep any unsused offsets at the current slope?
            eqns = np.delete(eqns,eqnset1,axis=0)
            # eqns = np.delete(eqns,eqnset1a,axis=0)
            # eqns = np.delete(eqns,eqnset1b,axis=0)

        avglines = np.reshape(avglines,(len(meqs)/2,4))
        meqs = np.reshape(meqs,(len(meqs)/2,2))
        # identificaton of tubes
        # target set of angles: head tube, seat tube, down tube, top tube  
        # conventional bike angles rotating from neg x to pos y are: (69,72,-47,-23)
        targetAngles = np.array([68,73,-47,-23])
        # corresponding target slopes of line segments:
        targetSlopes = np.tan(targetAngles * np.pi/180)
        dtL = avglines[np.abs(meqs[:,0] - targetSlopes[2]).argmin()]
        ttL = avglines[np.abs(meqs[:,0] - targetSlopes[3]).argmin()]
        stL = avglines[np.abs(meqs[:,0] - targetSlopes[1]).argmin()]
        htL = avglines[np.abs(meqs[:,0] - targetSlopes[0]).argmin()]
        rL = (htL,stL,dtL,ttL)
        # plot raw lines detection
        # bw2 = np.copy(bw)
        for Lline in rL:
            # for some reason can't follow this syntax with the line array as did with tuples
            #for x1,y1,x2,y2 in np.nditer(Lline):
            #    cv2.line(aimg2,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.line(bw2,tuple(Lline[0:2].astype(int)),tuple(Lline[2:4].astype(int)),(255,0,0),2)
        plotFig(bw2)
    
        return(rL)

    else:

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

        # need to convert to linexy here
        return(bimg,aimg,rA,rR)
    

def houghCircles(bw):
    # this preblur helps get rid of the apparent line in the middle of a tube
    # due to the reflection of light
    bwblur = cv2.blur(bw,(5,5))
    # not sure about final param1,2 choices yet
    wheelsInner = cv2.HoughCircles(bwblur,cv2.HOUGH_GRADIENT,
        1,minDist=200,param1=70,param2=50,minRadius=40,maxRadius=120)
    wheelsOuter = cv2.HoughCircles(bwblur,cv2.HOUGH_GRADIENT,
        1,minDist=200,param1=50,param2=40,minRadius=80,maxRadius=200)
    wheels = np.concatenate((wheelsInner,wheelsOuter),axis=1)
    # if (wheels is not None):
    #     # wheels = np.uint16(np.around(wheels))
    #     for i in wheels[0,:]:
    #         cv2.circle(aimg,(i[0],i[1]),i[2],(0,255,0),2)
    #         cv2.circle(aimg,(i[0],i[1]),2,(0,0,255),3)P.imW
    # not sure of logic. is not None? 
    # if (circles!=None):
    # find chainring
    chainring = cv2.HoughCircles(bwblur,cv2.HOUGH_GRADIENT,
        1,minDist=200,param1=70,param2=50,minRadius=10,maxRadius=30)
    # if (chainring is not None):
    #     # chainring = np.uint16(np.around(chainring))
    #     for i in chainring[0,:]:
    #         cv2.circle(aimg,(i[0],i[1]),i[2],(0,255,0),2)
    #         cv2.circle(aimg,(i[0],i[1]),2,(0,0,255),3)
    return bwblur,wheels,chainring

def findHeadTube(img,headtube):
    global figNo
    rows,cols = np.shape(img)[0:2]
    (m,b) = pts2eq([headtube[0],headtube[1]])
    htA = np.arctan(m) * 180/np.pi
    # point of rotation 
    htx,hty = int(np.round(np.mean(headtube[:,0]))),int(np.round(np.mean(headtube[:,1])))
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
    toprange = range(int(headtube[1,1])-5,int(headtube[1,1])-20,-1)
    # map search range from pixel units to the interpolated 0.1 pixel scale
    toprangeint = range(np.where(bxint==toprange[0])[0][0],np.where(bxint==toprange[-1])[0][0],-1)
    botrange = range(int(headtube[0,1])+5,int(headtube[0,1]+20))
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
 
    headtubetoplength = (headtube[1,1] - bxint[toppeak])
    headtubebotlength = (bxint[botpeak] - headtube[0,1])
    headtube[0] += np.array([np.cos(htA*np.pi/180)*headtubebotlength,np.sin(htA*np.pi/180)*headtubebotlength])
    headtube[1] -= np.array([np.cos(htA*np.pi/180)*headtubetoplength,np.sin(htA*np.pi/180)*headtubetoplength])
    return(headtube)

def calcOffset(t):
    # need to ensure all tubes are defined left/right. or the tubeset class has the hub points as well
    (m,b) = pts2eq([t[1][0],t[1][1]])
    htA = np.arctan(m) 
    forkLength = np.sqrt(pow(t[4][1][0]-t[4][0][0],2)+pow(t[4][1][1]-t[4][0][1],2)) 
    offsetAngle = np.arcsin( (t[4][1][0]-t[4][0][0]) / forkLength) + (htA-np.pi/2)
    offset = forkLength * np.sin(offsetAngle)
    return(offset)

def calcTrail(t,w):
    (m,b) = pts2eq([t[1][0],t[1][1]])
    htA = np.arctan(m) 
    wheelRadius = np.mean(w[0,2:4,2])
    # top of head tube is currently point 1 instead of 0
    trail = (wheelRadius+t[4][1][1]-t[1][1][1]) / np.sin(htA) * np.cos(htA) - (t[4][1][0]-t[1][1][0])
    return(trail)

def calcStandover(t,w):
    wheelRadius = np.mean(w[0,2:4,2])
    (m,b) = pts2eq([t[3][0],t[3][1]])
    standover = t[0][0][1] - (m * t[0][0][0] + b) + wheelRadius
    return(standover) 

if __name__=='__main__':
    filename = sys.argv[1]
    P = profilePhoto(filename)
    # aimg2 = np.copy(aimg)
    bimg,wheels,chainring = houghCircles(P.imGRAY)
    # remove wheels from gray image to unclutter for line detect
    P.imW = np.copy(P.imGRAY)
    P.imW = cv2.blur(P.imW,(5,5))
    cv2.circle(P.imW,(wheels[0,0,0],wheels[0,0,1]),int(wheels[0,0,2]+25.),255,-1)
    cv2.circle(P.imW,(wheels[0,1,0],wheels[0,1,1]),int(wheels[0,1,2]+25.),255,-1)
    cv2.circle(P.imW,(chainring[0,0,0],chainring[0,0,1]),int(chainring[0,0,2]+5),255,-1)

    # create working image
    # P.imW = np.copy(P.imGRAY)
    # P.imW = cv2.blur(P.imW,(5,5))
    # [0]have an extra dimension to get rid of here... also not able to reproduce head tube detection
    # results of hard-coded masking above. very sensitive somehow.
    # P.maskCircle(np.concatenate((wheels,chainring),axis=1)[0],P.imW)

    # built cv2 for imshow GTK UI support
    # but waitKey and destroyAllWindows are clunky use matplotlib for now
    # cv2.startWindowThread()
    # cv2.namedWindow('Circles Image')
    # cv2.imshow('Circles Image',houghCircleImage)
    # cv2.waitKey(1)
    #cv2.destroyAllWindows()
    # modified this to return line coords instead of rho/theta normals
    tubelines = houghLines(P.imW)
    tubeset = calcTubes2(tubelines,wheels,chainring)
    # improve head tube detection
    aimg2 = np.copy(P.imRGB)
    tubeset[1] = findHeadTube(aimg2,tubeset[1])
    # add fork, chainstay, seatstay
    tubeset.append( np.array([tubeset[1][0],wheels[0][0][0:2]]) )
    tubeset.append( np.array([wheels[0][1][0:2],chainring[0][0][0:2]]) )
    tubeset.append( np.array([wheels[0][1][0:2],tubeset[0][1]]) )

    offset = calcOffset(tubeset)
    print('offset =',offset)
    trail = calcTrail(tubeset,wheels)
    print('trail = ',trail)
    standover = calcStandover(tubeset,wheels)
    print('standover = ',standover)
    aimg2 = np.copy(P.imRGB)
    aimg2 = plotTubes(aimg2,tubeset)
    plotFig(aimg2,True)

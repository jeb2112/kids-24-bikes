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

def plotTubes(aimg,tubeset):
    for pt1,pt2 in tubeset:
        cv2.line(aimg,tuple(pt1.astype(int)),tuple(pt2.astype(int)),(0,0,255),2)
    return(aimg)

# redo function with just line segments in stead of normal lines
# messy float/int/array/tuple problem. cv2 functions work with tuples of ints, but coordinate
# calculations have to be arrays of floats. because only arrays can be cast back to ints for cv2.
# coordinates of each tube in x1,y1,x2,y2
# origin is cv2 top left
def calcTubes2(tlines,wheels,chainrings):
    frontwheel = np.array([wheels[0,0,0],wheels[0,1,1]])
    # need to check order of wheels
    backwheel = wheels[0,1,0:2]
    bottombracket = chainring[0,0,0:2]




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
    return (chainstay,toptube,seattube,forkhead)
    

def pts2eq(((x1,y1),(x2,y2))):
    m = float(y2-y1)/float(x2-x1)
    b = y1 - m * x1
    return [m,b]

def eq2pts((m,b),(x1,x2)):
    y1 = m*x1 + b
    y2 = m*x2 + b
    return ([x1,y1],[x2,y2])

def plotFig(img):
    global figNo
    plt.figure(figNo)
    plt.imshow(img)
    plt.show(block=False)
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

def houghLines(bimg,aimg):
    global figNo
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = bimg
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    for i in range(0,2):
        gray = cv2.dilate(gray,kernel,iterations=3)
        gray = cv2.erode(gray,kernel,iterations=2)

    edges = cv2.Canny(gray,100,200,apertureSize=3,L2gradient=True)
    plotFig(edges)

    # line processing
    houghProcess='p'
    if houghProcess=='p':
    # probabilistic works better for short segeements like head tube. these numbers are all fine tuned
    # and hard-coded for cleary though.
        lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=50)
        if (lines is not None):
            # plot raw lines detection
            aimg2 = np.copy(aimg)
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(aimg2,(x1,y1),(x2,y2),(0,0,255),2)
            plotFig(aimg2)

        # average matching pairs of lines
        eqns = np.zeros((np.shape(lines)[0],2))
        for i,line in enumerate(lines):
            eqns[i,:] = pts2eq(((line[0,0:2]),(line[0,2:4])))

        eq=None
        avglines=None
        while len(eqns)>0:
            eqnset = np.where((np.abs(eqns[:,0]-eqns[0,0])<.02)) and np.where((np.abs(eqns[:,1]-eqns[0,1])<12))
            if eq is None:
                # need to check for equal sets of pairs if eqnset is odd length for better average
                eq = np.reshape(np.mean(eqns[eqnset],axis=0),(1,2))
                avglines = eq2pts(eq[0],(0,600))
            else:
                eq2 = np.reshape(np.mean(eqns[eqnset],axis=0),(1,2))
                eq = np.append(eq,eq2,axis=0)
                avglines = np.append(avglines,eq2pts(eq2[0],(0,600)),axis=0)
            eqns = np.delete(eqns,eqnset,axis=0)

        avglines = np.reshape(avglines,(len(eq),4))
        # identificaton of tubes
        # target set of angles: head tube, seat tube, down tube, top tube  
        # conventional bike angles rotating from neg x to pos y are: (69,72,-47,-23)
        targetAngles = np.array([68,73,-47,-23])
        # corresponding target slopes of line segments:
        targetSlopes = np.tan(targetAngles * np.pi/180)
        dtL = avglines[np.abs(eq[:,0] - targetSlopes[2]).argmin()]
        ttL = avglines[np.abs(eq[:,0] - targetSlopes[3]).argmin()]
        stL = avglines[np.abs(eq[:,0] - targetSlopes[1]).argmin()]
        htL = avglines[np.abs(eq[:,0] - targetSlopes[0]).argmin()]
        rL = (htL,stL,dtL,ttL)
        # plot raw lines detection
        aimg2 = np.copy(aimg)
        for Lline in rL:
            # for some reason can't follow this syntax with the line array as did with tuples
            #for x1,y1,x2,y2 in np.nditer(Lline):
            #    cv2.line(aimg2,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.line(aimg2,tuple(Lline[0:2].astype(int)),tuple(Lline[2:4].astype(int)),(0,0,255),2)
        plotFig(aimg2)
    
        return(bimg,aimg,rL)

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
    

def houghCircles(bimg,aimg):
    # this preblur helps get rid of the apparent line in the middle of a tube
    # due to the reflection of light
    bimg = cv2.blur(bimg,(5,5))
    wheels = cv2.HoughCircles(bimg,cv2.HOUGH_GRADIENT,
        1,minDist=200,param1=70,param2=50,minRadius=60,maxRadius=140)
    if (wheels is not None):
        # wheels = np.uint16(np.around(wheels))
        for i in wheels[0,:]:
            cv2.circle(aimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(aimg,(i[0],i[1]),2,(0,0,255),3)
    # not sure of logic. is not None? 
    # if (circles!=None):
    # find chainring
    chainring = cv2.HoughCircles(bimg,cv2.HOUGH_GRADIENT,
        1,minDist=200,param1=70,param2=50,minRadius=10,maxRadius=30)
    if (chainring is not None):
        # chainring = np.uint16(np.around(chainring))
        for i in chainring[0,:]:
            cv2.circle(aimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(aimg,(i[0],i[1]),2,(0,0,255),3)
    return bimg,aimg,wheels,chainring

if __name__=='__main__':
    filename = sys.argv[1]
    im = imageio.imread(filename)
    # interesting note. the .png file has alpha channel.
    # suing cv2.line on rgba image, gives only white.
    # have to remove the alpha to get the rgb color for line
    aimg = cv2.cvtColor(im,cv2.COLOR_BGRA2BGR)
    # python copy reference
    # aimg2 = cv2.cvtColor(im,cv2.COLOR_BGRA2BGR)
    aimg2 = np.copy(aimg)
    print(aimg.shape)
    bimg = cv2.cvtColor(aimg,cv2.COLOR_BGR2GRAY)
    bimg,aimg2,wheels,chainring = (houghCircles(bimg,aimg2))
    # remove wheels from gray image to unclutter for line detect
    cv2.circle(bimg,(wheels[0,0,0],wheels[0,0,1]),int(wheels[0,0,2]+25.),255,-1)
    cv2.circle(bimg,(wheels[0,1,0],wheels[0,1,1]),int(wheels[0,1,2]+25.),255,-1)
    cv2.circle(bimg,(chainring[0,0,0],chainring[0,0,1]),int(chainring[0,0,2]+5),255,-1)

    # built cv2 for imshow GTK UI support
    # but waitKey and destroyAllWindows are so clunky why bother use matplotlib for now
    # cv2.startWindowThread()
    # cv2.namedWindow('Circles Image')
    # cv2.imshow('Circles Image',houghCircleImage)
    # cv2.waitKey(1)
    #cv2.destroyAllWindows()
    # modified this to return line coords instead of rho/theta normals
    bimg,aimg2,tubelines = houghLines(bimg,aimg2)
    plt.figure(figNo)
    figNo += 1
    blockTF=True
    plt.imshow(aimg2)
    plt.show(block=blockTF)
    # tubeset = calcTubes2(tubelines,wheels,chainring)
    # plotTubes(aimg,tubeset)

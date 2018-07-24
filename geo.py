import os
import sys
import io
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
import peakutils
import scipy.interpolate
import copy

figNo = 1
mmpx = 0
rows = 0 
cols = 0
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

class profilePhoto():
    global rows,cols
    def __init__(self,filename,mmpx):
        rows = 0
        cols = 0
        self.mmpx = mmpx # mm per pixel 
        self.filename = filename
        self.imRGB = None
        self.imGRAY = None
        self.imEDGE = None
        self.imANNO = None
        self.imW = None
        self.bg = 255
        self.loadImage(self.filename)

    def CM2PX(self,cm):
        return(int(np.round(cm/(self.mmpx/10.))))

    def loadImage(self,filename):
        global rows,cols
        im = imageio.imread(filename)
        # if png/alpha, remove alpha
        rows,cols = np.shape(im)[0:2]
        if np.shape(im)[2]==4:
            im = cv2.cvtColor(im,cv2.COLOR_BGRA2BGR)
        # may need this reverse?
        # else:
        #     im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        self.imRGB = im 
        self.imANNO = self.imRGB
        self.imGRAY = cv2.cvtColor(self.imRGB,cv2.COLOR_BGR2GRAY)
        self.bg = self.imGRAY[0,0]

    def houghCircles(self,bw):
        # this preblur maybe helps get rid of the apparent line in the middle of a tube
        # due to the reflection of light but this hasn't been investigated much yet
        # cleary,cujo24,AceLTD up to charger
        # bw1 = cv2.blur(bw,(5,5))
        # AceLTD. merely converting the hard-coded 5x5 kernel from cleary to same equivalent size broke it
        # reverting to 5x5, still had more difficulty with this image and cujo or cleary for the circles
        # fluid. switch to separate blurring for wheel/chainring. better tread outer detection this way
        bw1 = cv2.blur(bw,(CM2PX(1.6),CM2PX(1.6)))
        bw2 = cv2.blur(bw,(5,5))
        bw3 = np.copy(bw)
        # try simply binarizing the image? seemed like a good idea but param2 had to get even smaller to detect
        # anything, and the detections were anywhere but the correct spot. maybe need to reblur again?
        # ret,bw = cv2.threshold(bw,240,255,cv2.THRESH_BINARY)
        # plotFig(bw,False,cmap="gray")
        # plt.show()        
        # not sure about final param1,2 choices yet
        # cleary,cujo24
        # wheelsInner = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1,minDist=self.CM2PX(60),param1=self.CM2PX(22),param2=self.CM2PX(16),minRadius=self.CM2PX(13),maxRadius=self.CM2PX(30))
        # AceLTD...Alite-24
        # wheelsInner = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1.2,minDist=self.CM2PX(90),param1=self.CM2PX(10),param2=self.CM2PX(2),minRadius=self.CM2PX(20),maxRadius=self.CM2PX(30))        
        # Creig-24. had to downsample image out of memory, have to scale mmpx accordingly though 0.803 now
        # wheelsInner = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.2,minDist=self.CM2PX(90),param1=self.CM2PX(10),param2=self.CM2PX(2),minRadius=self.CM2PX(20),maxRadius=self.CM2PX(30))        
        # DYNAMITE_24. riprock
        # wheelsInner = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.2,minDist=self.CM2PX(90),param1=self.CM2PX(8),param2=self.CM2PX(2),minRadius=self.CM2PX(20),maxRadius=self.CM2PX(30))        
        # exceed. ewoc
        # wheelsInner = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.2,minDist=self.CM2PX(90),param1=self.CM2PX(4),param2=self.CM2PX(2),minRadius=self.CM2PX(20),maxRadius=self.CM2PX(30))        
        # frog62. never did pick up  the front inner correctly.
        # wheelsInner = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.1,minDist=self.CM2PX(90),param1=self.CM2PX(3),param2=self.CM2PX(2),minRadius=self.CM2PX(20),maxRadius=self.CM2PX(30))        
        # zulu. not picking up very well
        wheelsInner = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.1,minDist=self.CM2PX(86),param1=self.CM2PX(2),param2=self.CM2PX(2),minRadius=self.CM2PX(20),maxRadius=self.CM2PX(28))        
        # mantra. further reduction in maxRadius
        # wheelsInner = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.2,minDist=self.CM2PX(90),param1=self.CM2PX(8),param2=self.CM2PX(2),minRadius=self.CM2PX(20),maxRadius=self.CM2PX(26))        
        # pineridge. a scaling factor error was confusing the fit here, may not need unique params
        # wheelsInner = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.1,minDist=self.CM2PX(90),param1=self.CM2PX(4),param2=self.CM2PX(2),minRadius=self.CM2PX(20),maxRadius=self.CM2PX(24))        
        # place rear wheel first in list. note extra 1st dimension in output of HoughCircles
        wheelsInner = wheelsInner[0,wheelsInner[0,:,0].argsort(),:]
        # cleary.png
        # wheelsOuter = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1,minDist=self.CM2PX(60),param1=self.CM2PX(16),param2=self.CM2PX(13),minRadius=self.CM2PX(26),maxRadius=self.CM2PX(60))
        # cujo24.png
        # wheelsOuter = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1.2,minDist=self.CM2PX(60),param1=self.CM2PX(10),param2=self.CM2PX(6),minRadius=self.CM2PX(26),maxRadius=self.CM2PX(36))
        # AceLTD 1.2 seemed to make a big difference compared to 1.0? or was it dropping param1 way down to 4
        # wheelsOuter = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1.2,minDist=self.CM2PX(90),param1=self.CM2PX(4),param2=self.CM2PX(2),minRadius=self.CM2PX(26),maxRadius=self.CM2PX(40))
        # Bayview. further drop of param1 down to 3 required. that fixed the outerwheels, but lost the headtube!
        # wheelsOuter = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.2,minDist=self.CM2PX(90),param1=self.CM2PX(3),param2=self.CM2PX(2),minRadius=self.CM2PX(26),maxRadius=self.CM2PX(40))
        # pineridge. 
        # wheelsOuter = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.1,minDist=self.CM2PX(90),param1=self.CM2PX(4),param2=self.CM2PX(2),minRadius=self.CM2PX(28),maxRadius=self.CM2PX(40))
        # signal. not quite there.
        wheelsOuter = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.2,minDist=self.CM2PX(90),param1=self.CM2PX(3),param2=self.CM2PX(2),minRadius=self.CM2PX(28),maxRadius=self.CM2PX(40))
        wheelsOuter = wheelsOuter[0,wheelsOuter[0,:,0].argsort(),:]
        # argsort indexing removed dummy 1st dimension 
        wheels = np.concatenate((wheelsInner,wheelsOuter),axis=0)
        # cleary,cujo24
        # chainring = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,
        #     1,minDist=self.CM2PX(60),param1=self.CM2PX(22),param2=self.CM2PX(16),minRadius=self.CM2PX(3),maxRadius=self.CM2PX(10))[0]
        # AceLTD
        # chainring = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1,minDist=self.CM2PX(60),param1=self.CM2PX(10),param2=self.CM2PX(6),minRadius=self.CM2PX(3),maxRadius=self.CM2PX(10))[0]
        # alite-24. this reduced minDist detects couple dozen, to pick up the chainring.
        # chainring = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1,minDist=self.CM2PX(20),param1=self.CM2PX(4),param2=self.CM2PX(2),minRadius=self.CM2PX(3),maxRadius=self.CM2PX(10))[0]
        # fluid - didn't pick up the chairing or outer diameter properly
        # mantra - chainring detection with these params was skewed about 1cm high
        # chainring = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1,minDist=self.CM2PX(20),param1=self.CM2PX(3),param2=self.CM2PX(2),minRadius=self.CM2PX(3),maxRadius=self.CM2PX(8))[0]
        # pineridge. line. have to reduce maxradius because of false selection in mid-air above bottom bracket. could also mask that region out better to 
        # retain the large radius.
        chainring = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1,minDist=self.CM2PX(20),param1=self.CM2PX(3),param2=self.CM2PX(2),minRadius=self.CM2PX(3),maxRadius=self.CM2PX(6))[0]
        # BAYVIEW. use wheel hubs to select chainring circle of more than 1 detected
        if len(chainring[0])>1:
            wx,wy = np.mean(wheels[:,0:2],axis=0)
            chainring = np.reshape(chainring[(np.sqrt(pow(np.abs(chainring[:,0]-wx),2)+pow(np.abs(chainring[:,1]-wy),2))).argmin()],(1,3))
        for c in wheels:
            cv2.circle(bw3,(c[0],c[1]),int(c[2]),140,5)
        wc = np.mean([wheelsInner[:,0:2],wheelsOuter[:,0:2]],axis=0)   
        for w in wheels:
            cv2.circle(bw3,(w[0],w[1]),CM2PX(0.1),140,-1)
        for c in chainring:
            cv2.circle(bw3,(c[0],c[1]),int(c[2]),140,5)
        plotFig(bw3,False,cmap="gray",title='houghCircle: wheel detection')
        plt.show(block = not __debug__)        
        
        return wheels,chainring[0]

    def maskCircle(self,masks,target):
        # masks. list of circles defined by centre point, radius
        # target. target image
        # have option for extra pixels here, as the fits returned by hough were inordinately sensitive 
        # to the radius of the mask so further test needed. 
        # since headtube method improved should not need this anymore
        # AceLTD 2cm
        # BAYVIEW 3cm requried to get the headtube
        for c in masks:
            cv2.circle(target,(int(c[0]),int(c[1])),int(c[2]+self.CM2PX(0)),255,-1)        

    def maskRect(self,masks,target):
        # masks. list of rects defined by top left point, bottom right
        # target. target image
        for c in masks:
            cv2.rectangle(target,(c[0],c[1]),(c[2],c[3]),255,-1)        
        
    def selectCircle(self,maskcentre,maskradius,target):
        t2 = np.copy(target)
        cv2.circle(target,(maskcentre[0],maskcentre[1]),int(maskradius+self.CM2PX(0)),255,-1)        
        target = t2 - target

    def resetAnnotatedIm(self):
        self.imANNO = self.imRGB

    # single lines detection. this is probably what houghLines should be. line selection and combination
    # should be in a separate function
    def houghLinesS(self,bw,edgeprocess='bike',minlength=8.5):
        global figNo
        # bw = cv2.blur(bw,(5,5))
        # this sort of morphing might help flatten out smaller details like spokes and cables?? needs work
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        for i in range(0,2):
            bw = cv2.dilate(bw,kernel,iterations=3)
            bw = cv2.erode(bw,kernel,iterations=2)

        # cleary, cujo24, AceLTD
        # edges = cv2.Canny(bw,100,200,apertureSize=3,L2gradient=True)
        # BAYVIEW - reduce slightly due to lightyellow graphic that doesn't generate enough gradient in gray-scale
        if edgeprocess=='bike':
            edges = cv2.Canny(bw,95,190,apertureSize=3,L2gradient=True)
        # for wheel only
        elif edgeprocess=='fork':
            edges = cv2.Canny(bw,150,200,apertureSize=3,L2gradient=True)
        elif edgeprocess == 'head':
            edges = cv2.Canny(bw,150,200,apertureSize=3,L2gradient=True)
        plotFig(edges,False,title="houghLinesS")
        plt.show(block= not __debug__)
        # plotFigCV(edges)
        # plt.show(block=__debug__)

        lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=CM2PX(minlength))

        # todo. arrange all lines with pt1[0]<pt2[0]

        return(lines)


    # main lines detection
    def houghLines(self,bw,aw,houghProcess='p',edgeprocess='bike',minlength=8.5):
        global figNo
        # bw = cv2.blur(bw,(5,5))
        # this sort of morphing might help flatten out smaller details like spokes and cables?? needs work
        # metaHT. spokes are already removed, and the fit to the fork edges is a bit rough. 
        if edgeprocess=='bike':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            for i in range(0,2):
                bw = cv2.dilate(bw,kernel,iterations=3)
                bw = cv2.erode(bw,kernel,iterations=2)

        # cleary, cujo24, AceLTD
        # edges = cv2.Canny(bw,100,200,apertureSize=3,L2gradient=True)
        # BAYVIEW - reduce slightly due to lightyellow graphic that doesn't generate enough gradient in gray-scale
        # ewoc - bg=236. changed to 255 in gimp, but could change all the hard-coded fills here to P.bg
        if edgeprocess=='bike':
            if self.bg > 200:
                # works. revert to small aperture
                edges = cv2.Canny(bw,95,190,apertureSize=3,L2gradient=True)
                # cube240... trail. for the bottom side of top tube which was obscured by cable
                # edges = cv2.Canny(bw,95,190,apertureSize=7,L2gradient=True)
        # filtering on the black background didn't work, had to use gimp
            elif self.bg == 0:
                edges = cv2.Canny(bw,1,2,apertureSize=7,L2gradient=True)
        # for wheel only
        else:
            edges = cv2.Canny(bw,150,200,apertureSize=3,L2gradient=True)
        plotFig(edges,False,title='houghLines: edges')
        # plotFigCV(edges)
        plt.show(block=  not __debug__)

        # line processing
        if houghProcess=='p':
        # probabilistic works better for short segments like head tube.
        # 1. estimate of headset might be better by clipping the rigid fork exacttly at the headtube. or for suspension,
        # using the fork too.
        # 2. average line for the downtube in biased for the cleary example, can't see why the averaing doesn't worok
            # cleary lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=CM2PX(15.75))
            # AceLTD-Alite-24
            # lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=CM2PX(8.5))
            # Creig-24 seem to miss the headtube so try 6.5. and stxdt at bottom bracket could be constrained by chainring
            lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=CM2PX(minlength))
            if (lines is not None):
                # plot raw lines detection
                bw2 = np.copy(aw)
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(bw2,(x1,y1),(x2,y2),(0,0,255),CM2PX(0.2))
                plotFig(bw2,False,title='houghLines: raw lines')
                plt.show(block= not __debug__)
            else:
                print('No edges detected')

            # average matching pairs of lines. slope/intercept might not be ideal to select the matching pairs
            # becuase non-linear.
            eqns = np.zeros((np.shape(lines)[0],2))
            rhotheta = np.zeros((np.shape(lines)[0],2))
            for i,line in enumerate(lines):
                eqns[i,:] = pts2eq(((line[0,0:2]),(line[0,2:4])))
                rhotheta[i,:] = coord2angle(line)

            # remove any with m=0 or NaN. nete extra dimension in l ines
            setr = [a and b for a,b in zip(np.abs(eqns[:,0])>0.01,~np.isnan(eqns[:,0]))] 
            lines = lines[setr,:,:]
            eqns = eqns[setr,:]
            rhotheta = rhotheta[setr,:]

            # cujo24.png. saddle is partially picked up. from photo cujo saddle appears to be an adult size 26cm
            # can add a check for saddle length? saddle length isn't tabulated by vendors
            meqs=None
            avglines=None
            while len(eqns)>0:
                # find all equations matching the slope of current first eqn in list. Using 15% as the matching threshold
                eqnset1 = np.where(np.abs(eqns[:,0]-eqns[0,0])<np.abs(.15*eqns[0,0]))[0][0:]
                # for debugging: plot raw lines detection
                bw2 = np.copy(aw)
                for line in lines[eqnset1]:
                    for x1,y1,x2,y2 in line:
                        cv2.line(bw2,(x1,y1),(x2,y2),(255,0,255),2)
                # disply individual eqn sets for debugging purpose
                # plotFig(bw2)
                # plt.show()
                # # equal slope, equal offset. Using 2% to qualify as equal.
                # this logic may still work but no good for tapered tubes. cujo24, metaHT
                # eqnset1a =  eqnset1[np.where((np.abs(eqns[eqnset1,1]-eqns[0,1])<np.abs(0.01*eqns[0,1])))]
                # alite-24 - increase it back up to 2%. detecting too many false lines though, have to select better
                # eqnset1a =  eqnset1[np.where((np.abs(eqns[eqnset1,1]-eqns[0,1])<np.abs(0.02*eqns[0,1])))]
                # metaHT. tapered top tube throws off this logic. using rhotheta with smaller threshold but need overhaul
                # eqnset1a =  eqnset1[np.where((np.abs(rhotheta[eqnset1,0]-rhotheta[0,0])<np.abs(0.01*rhotheta[0,0])))]
                # mxxc increase threshold.... to pineridge. yamajama24
                eqnset1a =  eqnset1[np.where((np.abs(rhotheta[eqnset1,0]-rhotheta[0,0])<np.abs(0.015*rhotheta[0,0])))]
                # equal slope, different offset but close enough to be a matchingline
                # y intercept 10% as the threshold%. 10% too low. try 20%.
                # if change to rhotheta then can identify by a length in actual pixels
                eqnset1b = np.setdiff1d(eqnset1,eqnset1a)
                eqnset1b = eqnset1b[np.where(np.abs(eqns[eqnset1b,1]-eqns[0,1])<np.abs(0.2*eqns[0,1]))]
                # equal slope, different offset
                if len(eqnset1a) == 0:
                    print('No equations in set1a')
                    break
                else:
                    meq1 = np.mean(eqns[eqnset1a],axis=0)

                if len(eqnset1b) > 0:
                    meq2 = np.mean(eqns[eqnset1b],axis=0)
                    meq = np.mean([meq1,meq2],axis=0)
                else:
                    # hough has failed to create a pair of edgelines in the near vicinity.
                    # 2nd pass attempt using profile to find a matching line in case it is one of the main tubes
                    if edgeprocess=='bike':
                        # form profile across the existing edgeline and search for edges along that profile
                        mln1 = np.mean(lines[eqnset1a,:],axis=0)
                        t=Tube()
                        t.setpts(mln1[0,0:2],mln1[0,2:4])
                        midpt = np.mean((t.pt1,t.pt2),axis=0)
                        p = Profile(int(midpt[0]),int(midpt[1]),CM2PX(5),CM2PX(5),RAD2DEG(t.A)-90,self.imW)
                        p.setprofile()
                        try:
                            p.setpeaks1()
                            p.setwidth()
                            p.setedge()
                            p.hcentre = p.prof2pix((p.htx2,midpt[1]))[:,0]
                            # adjust rho according to the 2nd line of the pair
                            # keep track of sign here
                            t.setrhotheta(t.rho-(p.hcentre[0]-p.hedge[0]),t.theta)
                            meq = np.array([t.m,t.b])
                        except profileException:
                            print("Peaks not detected correctly")
                            meq = np.array([0,0])
                            
                    # not using this logic for the fork detection yet
                    elif edgeprocess=='fork':
                        meq = meq1

                if meqs is None:
                    meqs = meq
                    avglines = eq2pts(meq,(0,cols))
                else:
                    meqs = np.append(meqs,meq,axis=0)
                    avglines = np.append(avglines,eq2pts(meq,(0,cols)),axis=0)
                # throw out only the used offsets 
                eqns = np.delete(eqns,np.concatenate((eqnset1a,eqnset1b),axis=0),axis=0)
                rhotheta = np.delete(rhotheta,np.concatenate((eqnset1a,eqnset1b),axis=0),axis=0)
                lines = np.delete(lines,np.concatenate((eqnset1a,eqnset1b),axis=0),axis=0)

            avglines = np.reshape(avglines,(len(meqs)/2,4))
            meqs = np.reshape(meqs,(len(meqs)/2,2))
        
            return(avglines,meqs)

    def findForkLines(self,wheel,minlength):
        # try to fill in light-colored paint with a preliminary thresh. hardcoded at 250 for now
        imW = np.copy(self.imW)
        if self.bg > 250:
            ret,imW = cv2.threshold(imW,250,255,cv2.THRESH_BINARY)
        elif self.bg == 0:
            ret,imW = cv2.threshold(imW,250,255,cv2.THRESH_BINARY)
        # im = cv2.blur(im,(5,5))
        # Creig-24 . try more blur more spoke suppression, suppression of graphic/printing
        imW = cv2.blur(imW,(15,15))
        imW2 = np.copy(imW)
        self.selectCircle(wheel.centre,wheel.rOuter,imW)
        imW = imW2 - imW
        # 100 threshold helps get rid of spokes
        ret,imW = cv2.threshold(imW,100,255,cv2.THRESH_TOZERO_INV)

        plotFig(imW,cmap="gray")
        plt.show(block= not __debug__)
        # up to Creig-24. charger
        # avglines,meqs = P.houghLines(imW,P.imRGB,edgeprocess='fork',minlength=7.5)
        # fluid had to reduce minlength 
        avglines,meqs = self.houghLines(imW,self.imRGB,edgeprocess='fork',minlength=minlength)
        return(avglines,meqs)


###############
# class Profile
###############

class profileException(Exception):
    pass

# class for processing line profiles in image. for now the line profiles are obtained horizontally by
# specifying the angle of rotation of the source image accordingly. the class instances are intended
# to be used in iterations to find optimal values but this is only half-conceived.
class Profile():
    def __init__(self,hx,hy,hl,hr,A,img):
        # centre of rotation, angle, for source image to obtain horizontal profile
        self.hx = hx
        self.hy = hy
        self.A = A
        self.img = img
        # process rgb or gray
        if len(img.shape)==2:
            self.ndim=1
        elif len(img.shape)==3:
            self.ndim=3
        # pixel range horizontally for profile
        self.hrange = range(self.hx-hl,self.hx+hr)
        self.hprofile = np.zeros(len(self.hrange))
        self.hrangeline = np.reshape(np.array([self.hrange[0],self.hy,self.hrange[-1],self.hy]),(1,1,4))
        # arbitrary factor of 10 for interpolation
        self.bxint = np.round(np.arange(self.hrange[0],self.hrange[-1],.1)*10)/10
        self.bspl = np.zeros((len(self.bxint),3))  
        self.mbspl = np.zeros(len(self.bxint))       

        self.hpeaks = np.zeros((2,2))
        # two types of detection will be used and kept track of separately
        self.hpeaks1 = np.zeros(4)
        self.hpeaks2 = np.zeros(4)
        self.hwidth1 = 0
        self.hwidth2 = 0
        self.hwidth = 0
        self.hedge = np.zeros((2,2))
        self.hcentre = np.zeros((2,2))
        self.htx2 = np.zeros(2)
        self.hshift = np.zeros(2)

        self.setrimg()
        self.setprofile()

    # convert final width of the profile to a pair of edges and a centre point
    def setedge(self):
        if self.hwidth1 and self.hwidth2:
            if self.hwidth1 < self.hwidth2:
                self.hedge = self.bxint[self.hpeaks1[0:2].astype(int)]
            else:
                self.hedge = self.bxint[self.hpeaks2[0:2].astype(int)]
        elif self.hwidth1:
            self.hedge = self.bxint[self.hpeaks1[0:2].astype(int)]
        elif self.hwidth2:
            self.hedge = self.bxint[self.hpeaks2[0:2].astype(int)]
        # new centreline of the tube
        self.htx2 = np.mean(self.hedge,axis=0)

    # compare widths determined by two methods and choose the resultant width
    # the width refers to a tube across which the profile is taken so a misnomer 
    # for the 'length' of the profile.
    # the process for establishing the tube width is expected to be iterative so 
    # raises an exception if failure.
    def setwidth(self):
        if self.hwidth1 and self.hwidth2:
            self.hwidth = min(self.hwidth1,self.hwidth2)
        elif self.hwidth1 or self.hwidth2:
            self.hwidth = max(self.hwidth1,self.hwidth2)
        else:
            print("measureHeadTube: no width detected")
            raise profileException

    # second method for establishing tube width based on two adjacent peaks in the derivative on either side of the search point
    # min_dist criterion is therefore not used.
    def setpeaks2(self):
        peaks2 = peakutils.indexes(self.mbspl,thres=0.2,min_dist=CM2PX(0)*10)
        # two max peaks should be the main edges if not the only ones selected. may need threshold image here though
        # riprock,pineridge fails at pt1 due to narrow gap and brake
        # edge. with threshold image, can rely on two innermost peaks as the criterion. didn't work for xtcsljr
        if len(peaks2) >= 2:
            idx = np.searchsorted(peaks2,len(self.bxint)/2)
            self.hpeaks2[0:2] = peaks2[idx-1:idx+1]
            self.hwidth2 = self.bxint[self.hpeaks2[1].astype(int)]-self.bxint[self.hpeaks2[0].astype(int)]

    # first method for establishing tube width based on the spline derivative amplitude, edges should have the
    # largest derivative values, but min_dist criterion helps some borderline cases.
    def setpeaks1(self):
        peaks1 = peakutils.indexes(self.mbspl,thres=0.2,min_dist=CM2PX(3)*10)
        if len(peaks1)>=2:
            self.hpeaks1[0:2] = np.sort(peaks1[self.mbspl[peaks1].argsort()][::-1][0:2])
            self.hwidth1 = self.bxint[self.hpeaks1[1].astype(int)]-self.bxint[self.hpeaks1[0].astype(int)]
        else:
            raise profileException

    # rotate source image so desired profile is horizontal
    def setrimg(self):
            self.M = cv2.getRotationMatrix2D((self.hx,self.hy),self.A,1)
            self.Minv = np.concatenate((np.linalg.inv(self.M[:,0:2]),np.reshape(-np.matmul(np.linalg.inv(self.M[:,0:2]),self.M[:,2]),(2,1))),axis=1)
            self.rimg = cv2.warpAffine(self.img,self.M,(cols,rows))

    # rotate a point on the profile back to pixel coordinates
    def prof2pix(self,pt):
        newpt = np.matmul(self.Minv[:,0:2],np.reshape(pt,(2,1)))
        if np.shape(self.Minv)[1]==3:
            newpt += np.reshape(self.Minv[:,2],(2,1))
        return( newpt )

    # returns a smoothing spline of the first derivative of the profile used for detecting edges
    # for now rgb profiles are averaged rather than processed individually
    def setprofile(self):
        self.hprofile = self.rimg[self.hy,self.hrange]
        if self.ndim==1:
            self.hprofile = np.reshape(self.hprofile,(len(self.hprofile),1))

        for i in range(0,self.ndim):
            # arbitrary smoothing factor
            bsp = scipy.interpolate.splrep(self.hrange,self.hprofile[:,i],np.ones(len(self.hrange)),k=3,s=len(self.bxint))
            self.bspl[:,i] = scipy.interpolate.splev(self.bxint,bsp,der=1)
        # pineridge. need color to get the measurement due to black gear trigger and no white gap
        self.mbspl = np.mean(np.abs(self.bspl[:,0:self.ndim]),axis=1)


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
        self.m,self.b = pts2eq((pt1,pt2))
        self.A = np.arctan(self.m)
        # list() unarrays the np.array, while [] retains it
        self.rho,self.theta = coord2angle([np.concatenate((pt1,pt2),axis=0)])
        self.len = self.l()

    def seteqn(self,m,b):
        self.m = m
        self.b = b
        if self.pt2[0]==0:
            self.pt2[0]=cols
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
        self.m,self.b = pts2eq((self.pt1,self.pt2))
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



class Tubeset():
    def __init__(self):
        self.tubes = dict(dt=Tube(),tt=Tube(),st=Tube(),ht=Tube(),cs=Tube(),ss=Tube())
        self.targets = dict(ht=Tube(),st=Tube())
        # this will replace targetAngles, targetSLopes
        self.targets['ht'].A = DEG2RAD(70)
        self.targets['st'].A = DEG2RAD(72)
        # target set of angles: head tube, seat tube, down tube, top tube  
        # conventional bike angles rotating from neg x to pos y are: (69,72,-47,-23)
        # cujo24. downtube slope>-47 so it picked the stem angle. either filter out stem line before or lower target
        self.targetAngles = dict(ht=68,st=73,dt=-55,tt=-23)
        self.targetSlopes = dict()
        for key in self.targetAngles.keys():
            self.targetSlopes[key] = np.tan(self.targetAngles[key] * np.pi/180)

    # three main tubes can be assigned by slope alone
    def assignTubeLines(self,avglines,meqs,tubes):

        for tube in tubes:
            self.tubes[tube].setpts(avglines[np.abs(meqs[:,0] - self.targetSlopes[tube]).argmin()][0:2],
                                    avglines[np.abs(meqs[:,0] - self.targetSlopes[tube]).argmin()][2:4])

    # match closest line to head tube target. lines not checked yet for same convention pt1[0]<pt2[0]
    def assignHeadTubeLine(self,lines):
        md=np.zeros(len(lines))
        for i,line in enumerate(lines):
            line = line[0]
            d1 = np.sqrt(pow(line[0]-self.targets['ht'].pt1[0],2)+pow(line[1]-self.targets['ht'].pt1[1],2))
            d2 = np.sqrt(pow(line[2]-self.targets['ht'].pt2[0],2)+pow(line[3]-self.targets['ht'].pt2[1],2))
            md[i] = np.mean([d1,d2])
        selected = lines[md.argmin()][0]
        self.tubes['ht'].setpts(selected[0:2],selected[2:4])
        # adjust line by less than half the diamter of the expected headtube. 
        # this will ensure a clean profile for the actual headtube length detection, following
        # which the more accurate diameter detection is done. 
        # vertex. increase slightly.
        self.tubes['ht'].rho = self.tubes['ht'].rho - CM2PX(1.0*2.54/2)
        self.tubes['ht'].setrhotheta(self.tubes['ht'].rho,self.tubes['ht'].theta)

    # match closest lines to seat tube target. lines not checked yet for same convention pt1[0]<pt2[0]
    def assignSeatTubeLine(self,lines):
        # 1 line only. shift the line to pass through bottom bracket. not done yet.
        if len(lines)==1:
            self.tubes['st'].setpts(lines[0:2],lines[2:4])
            self.tubes['st'].rho = self.targets['st'].rho
            self.tubes['st'].setrhotheta(self.tubes['st'].rho,self.tubes['st'].theta)
        else:
            md=np.zeros(len(lines))
            for i,line in enumerate(lines):
                line = line[0] # have an extra dimension to get rid of here
                d1 = np.sqrt(pow(line[0]-self.targets['st'].pt1[0],2)+pow(line[1]-self.targets['st'].pt1[1],2))
                d2 = np.sqrt(pow(line[2]-self.targets['st'].pt2[0],2)+pow(line[3]-self.targets['st'].pt2[1],2))
                md[i] = np.mean([d1,d2])
            # sort lines on distance
            lines = lines[md.argsort()]
            md = md[md.argsort()]
            # iterate and accumulate lines for 1st or 2nd edge or break. possibly this should just use the 
            # first two lines from the sort, eg pineridge a 3rd line that is close throws off the result
            t=Tube()
            rt1 = np.zeros((1,2))
            rt2 = np.zeros((1,2))
            set1 = np.zeros((1,2))
            for i,line in enumerate(lines):
                line=line[0]
                t.setpts(line[0:2],line[2:4])
                # pineridge. increase to 5 cm. line increase to 5.5
                if np.abs(t.rho - self.targets['st'].rho) > CM2PX(5.5) or RAD2DEG(np.abs(t.theta - self.targets['st'].theta)) > 10:
                    break
                else:
                    set1 = np.concatenate((set1,np.reshape(np.array([t.rho,t.theta]),(1,2))),axis=0)
            # take mean angle for improved centreline approx
            meantheta = np.mean(set1[1:,1])
            # update the target. this allows to use rho to detect which edge is which below
            self.createSeatTubeTarget(cr=None,A=meantheta)
            # sort lines into edges for averaging
            set1 = set1[1:]
            # sethist,setbins = np.histogram(set1,range(int(np.amin(set1[:,0]))-CM2PX(0.5),int(np.amin(set1[:,0]))+CM2PX(5.5),CM2PX(1)))
            for rho,theta in set1:
                if rho < self.targets['st'].rho:
                    rt1 = np.concatenate((rt1,np.reshape(np.array([rho,theta]),(1,2))),axis=0)
                else:
                    rt2 = np.concatenate((rt2,np.reshape(np.array([rho,theta]),(1,2))),axis=0)
            if len(rt1)>1:
                rt1 = np.mean(rt1[1:],axis=0)
            else:
                rt1 = None
            if len(rt2) > 1:
                rt2 = np.mean(rt2[1:],axis=0)
            else:
                rt2 = None
            if rt1 is None and rt2 is None:
                print('assignSeatTubeLines: no targets matched')
                return
            if rt1 is not None and rt2 is not None:
                rt = np.mean((rt1,rt2),axis=0)
                self.tubes['st'].setrhotheta(rt[0],rt[1])
            else:
                if rt2 is None:
                    t.setrhotheta(rt1[0],rt1[1])
                    b = self.targets['st'].pt2[1] - t.m * self.targets['st'].pt2[0]
                    self.tubes['st'].seteqn(t.m,b)
                if rt1 is None:
                    t.setrhotheta(rt2[0],rt2[1])
                    b = self.targets['st'].pt2[1] - t.m * self.targets['st'].pt2[0]
                    self.tubes['st'].seteqn(t.m,b)


    def createHeadTubeTarget(self,fw,type='susp'):
        if type=='susp':
            travel = CM2PX(6.5)
        elif type=='rigid':
            travel = 0
        crown = CM2PX(4.5)
        length = CM2PX(10)

        axle2crown = fw.rOuter + (travel + crown)
        self.targets['ht'].pt1 = (fw.centre[0]-(axle2crown+length) * np.cos(self.targets['ht'].A),
                                fw.centre[1]- (axle2crown+length)*np.sin(self.targets['ht'].A))
        self.targets['ht'].pt2 = (fw.centre[0]-(axle2crown) * np.cos(self.targets['ht'].A),
                                fw.centre[1]- (axle2crown)*np.sin(self.targets['ht'].A))

    # try to mask only the seat, not the seatpost. add seatpost detection to this
    def createSeatTubeTarget(self,cr,A=None):
        length = CM2PX(12)
        if A is not None:
            self.targets['st'].A = A
            if cr is None and self.targets['st'].fixed is None:
                print('Seat tube not initialized')
                return 
            else:
                self.targets['st'].settheta(A)

        if cr is not None:
            pt1 = (cr.centre[0]-(cr.R + length) * np.cos(self.targets['st'].A),
                                    cr.centre[1]- (cr.R + length)*np.sin(self.targets['st'].A))
            pt2 = (cr.centre[0]-(cr.R) * np.cos(self.targets['st'].A),
                                    cr.centre[1]- (cr.R)*np.sin(self.targets['st'].A))
            self.targets['st'].fixed = cr.centre
            self.targets['st'].setpts(pt1,pt2)

    # average the slope of existing tube with correction 
    def modifyTubeLines(self,meqs,tube,op='mean'):
        # original code. update while retaining one of the points fixed. not very accurate.
        # not using existing ht slope in comparison here in case it is grossly wrong (ie the replace case)
        if len(meqs)>1:
            targ = np.abs(meqs[:,0] - self.targetSlopes[tube]).argmin()
            m2 = meqs[targ,0]
            if op=='mean':
                m2 = np.mean([self.tubes[tube].m,m2])
            elif op=='replace':
                pass
            # calculate new b modified line retaining existing point. could use average of pt1,pt2
            b2 = self.tubes[tube].pt1[1] - m2*self.tubes[tube].pt1[0]
        # new. combine with both slope and intercept and both points update. 
        else:
            m2 = meqs[0,0]
            b2 = meqs[0,1]
            if op=='mean':
                m2 = np.mean([self.tubes[tube].m,m2])
                b2 = np.mean([self.tubes[tube].b,b2])
            elif op=='replace':
                pass
        self.tubes[tube].seteqn(m2,b2)
        
    # float/int/array/tuple problem. cv2 functions work with tuples of ints, but coordinate
    # calculations have to be arrays of floats. because only arrays can be cast back to ints for cv2.
    # coordinates of each tube in x1,y1,x2,y2
    # origin is cv2 top left
    def calcTubes(self):
        # note convention is pt1 is lower value of x than pt2
        for t1,t2,pt1,pt2 in [('st','tt','pt1','pt1'),('tt','ht','pt2','pt1'),('dt','ht','pt2','pt2'),('dt','st','pt1','pt2')]:
            xint = ( self.tubes[t1].b-self.tubes[t2].b ) / ( self.tubes[t2].m - self.tubes[t1].m )
            setattr(self.tubes[t1],pt1,np.array([xint,self.tubes[t1].y(xint)]))
            setattr(self.tubes[t2],pt2,np.array([xint,self.tubes[t2].y(xint)]))
        # check if incorrect detection of curved tubes has created a non-physical tt/dt intersection
        # needs proper detection for line segments of a bent tube. quick kludge for now
        if self.tubes['tt'].pt2[1] - self.tubes['dt'].pt2[1] > -CM2PX(1):
            # kludge: alter the down tube to parallel the top tube. for straight top tube
            kludge=1
            if kludge==1:
                xint = ( self.tubes['tt'].b-self.tubes['dt'].b ) / ( self.tubes['dt'].m - self.tubes['tt'].m )
                # trail. decreased this hard-coded offset from 6 bback to 4
                self.tubes['dt'].pt2[0]= xint - CM2PX(4)
                self.tubes['dt'].pt2[1] = self.tubes['dt'].y(self.tubes['dt'].pt2[0])
                self.tubes['gt'] = Tube()
                # y intercept for gusset tube, using the point of truncation
                b = self.tubes['dt'].pt2[1] - self.tubes['tt'].m*self.tubes['dt'].pt2[0]
                self.tubes['gt'].seteqn(self.tubes['tt'].m,b)
                self.tubes['gt'].pt1 = np.array([xint-CM2PX(4),self.tubes['gt'].y(xint-CM2PX(4))])
                xint = ( self.tubes['ht'].b-self.tubes['gt'].b ) / ( self.tubes['gt'].m - self.tubes['ht'].m )
                self.tubes['gt'].pt2 = np.array([xint,self.tubes['gt'].y(xint)])
                # recalculate the head tubes point 2
                self.tubes['ht'].pt2 = np.array([xint,self.tubes['ht'].y(xint)])
            elif kludge==2:
            # kludge 2: swap the intersections points at the head tube. for bent top tube like riprock
                xint1 = ( self.tubes['ht'].b - self.tubes['tt'].b ) / ( self.tubes['tt'].m - self.tubes['ht'].m )
                y1 = self.tubes['tt'].y(xint1)
                xint2 = ( self.tubes['ht'].b - self.tubes['dt'].b ) / ( self.tubes['dt'].m - self.tubes['ht'].m )
                y2 = self.tubes['dt'].y(xint2)
                self.tubes['dt'].setpts(self.tubes['dt'].pt1,np.array([xint1,y1]))
                self.tubes['tt'].setpts(self.tubes['tt'].pt1,np.array([xint2,y2]))
        return

    # use first estimate of head tube to improve by measuring tube width at the top and bottom
    def measureHeadTube(self,img):
        global figNo
        p=[]
        meq = np.zeros((1,2))
        for i1 in range(0,2):
            if i1==0:
                htx,hty = self.tubes['ht'].pt1
            else:
                htx,hty = self.tubes['ht'].pt2
            p.append(Profile(int(htx),int(hty),CM2PX(5.5),CM2PX(3.0),RAD2DEG(self.tubes['ht'].A)-90,img))
            # rotate around the current estimate of the head tube point
            # M = cv2.getRotationMatrix2D((htx,hty),self.tubes['ht'].A*180/np.pi-90,1)
            # Minv = np.concatenate((np.linalg.inv(M[:,0:2]),np.reshape(-np.matmul(np.linalg.inv(M[:,0:2]),M[:,2]),(2,1))),axis=1)
            # rimg = cv2.warpAffine(img,M,(cols,rows))
            # needed 4 cm to the left for the wider headtube of riprock
            # hrange = range(int(htx)-CM2PX(4),int(htx)+CM2PX(3))
            # needed less than 3cm to right cube240
            # adjust again for mxxc 4.5cm to left
            # kato initial line very close to right tube edge. reduce range
            # can detect this properly based on teh 255 background
            # hrange = range(int(htx)-CM2PX(4.5),int(htx)+CM2PX(2))
            # yamajama - tapered head tube need more on the bottom. problem with top measure on the right from brakes exclude with 2.5 for now
            # zulu increase right-hand range slightly
            # debug only
            # rimg2=np.copy(rimg)
            # plotLines(rimg2,p[i1].hrangeline.astype(int),False,cmap="gray")

            # slide the profile up/down until a consistent width is detected. 
            hy = int(hty)
            hi = 0
            hwidthold  = 0.0
            p2=[]
            try:
                while True:
                    p2.append(copy.deepcopy(p[i1]))
                    p2[hi].hy = hy
                    p2[hi].setprofile()
                    p2[hi].setpeaks1()
                    p2[hi].setpeaks2()
                    p2[hi].setwidth()

                    if hwidthold == 0.0 or (p2[hi].hwidth - hwidthold < -0.01*hwidthold):
                        hwidthold = p2[hi].hwidth
                        hi += 1
                        hy += pow(-1,i1)
                    elif p2[hi].hwidth - hwidthold > 0.01*hwidthold:
                        print("measureHeadTube: converged")
                        hi -= 1
                        raise profileException
                    # or a large jump to increased width also cause to end iteration
                    else:
                        print("measureHeadTube: converged")
                        raise profileException

                    # not likely more than 1-2 cm of extended head tube.   
                    if hi==CM2PX(1):
                        print("measureHeadTube: ending iteration at 1 cm")
                        raise profileException

            except profileException:
                pass

            # find the edges at the converted width
            p2[hi].setedge()

            # if consistent width detected, rotate back to pixel coordinates. 
            p2[hi].hcentre = p2[hi].prof2pix((p2[hi].htx2,hty))[:,0]

            # save iteration result
            p[i1] =  copy.deepcopy(p2[hi])

        # plots
        plt.figure(figNo)
        for i1 in range(0,2):
            plt.subplot(2,1,i1+1)
            if i1==0:
                plt.title('measureHeadTube')
            plt.plot(p[i1].bxint,p[i1].mbspl)
            plt.plot(p[i1].hrange,p[i1].hprofile)
            plt.plot(p[i1].bxint[p[i1].hpeaks1[0:2].astype(int)],p[i1].mbspl[p[i1].hpeaks1[0:2].astype(int)],'r+')

        plt.show(block= not __debug__)
        figNo = figNo + 1

        # mxxc. revert to use of separate values to allow for the correction of the head tube angle
        # also check for positive error value. the bias to the right of the original head tube detection
        # should bias these values to negative. 
        h1shift = p[0].hcentre[0]-self.tubes['ht'].pt1[0]
        h2shift = p[1].hcentre[0]-self.tubes['ht'].pt2[0]
        if h1shift < CM2PX(0.5) and h2shift < CM2PX(0.5):
            meq[0,:] = pts2eq((p[0].hcentre,p[1].hcentre))
            # remove previous gusset tube if present
            if 'gt' in self.tubes.keys():
                self.tubes.pop('gt')
        else:
            print "measureHeadTube: no detection"
            meq=None
        return meq

    def extendHeadTube(self,img):
        global figNo
        # rows,cols = np.shape(img)[0:2]
        # point of rotation. could be the average of the 1st pass head tube pts, if they are evenly placed
        # but some times, the downtube will curve at the join, making the lower head tube pt2
        # artificially high. meanwhile the top tube will likely never curve at the join.
        # therefore probably should use only the top point, and skew the search range accordingly.
        htx,hty = np.round(np.mean([self.tubes['ht'].pt1,self.tubes['ht'].pt2],axis=0)).astype(int)
        M = cv2.getRotationMatrix2D((htx,hty),self.tubes['ht'].A*180/np.pi-90,1)
        rimg = cv2.warpAffine(img,M,(cols,rows))
        lrange = range(hty-CM2PX(10),hty+CM2PX(10))
        # signal. long head tube 
        # lrange = range(hty-CM2PX(10),hty+CM2PX(12))
        # AceLTD. the central profile overlapped some welds which broke the detection.
        # bias the profile to the right away from the seat/down gusset
        # htprofile = rimg[lrange,htx+CM2PX(0.2)]
        # mxtrail. all black. 1 profle alone disapperaed in shadow. try combining two or three
        # htprofile = (rimg[lrange,htx+CM2PX(0.2)] + rimg[lrange,htx+CM2PX(0.8)])/2     
        # mxxc. opposite problem. don't use a bias
        htprofile = rimg[lrange,htx+CM2PX(0.0)]
   
        # fit spline to colour profiles
        bxint = np.round(np.arange(lrange[0],lrange[-1],.1)*10)/10
        bspl = np.zeros((len(bxint),3))
        ypeaks=[]
        for i in range(0,3):
            # arbitrary smoothing factor
            # trail. subtle shadow adds 3rd peak to cable crossing. 
            bsp = scipy.interpolate.splrep(lrange,htprofile[:,i],np.ones(len(lrange)),k=3,s=len(bxint))
            bspl[:,i] = scipy.interpolate.splev(bxint,bsp,der=1)
            # decided not to use the indivdual colour traces for now
            # ypeaks.append(bxint[list(peakutils.indexes(bspl[:,i],thres=0.5,min_dist=CM2PX(30)))])

        # create detection search ranges. 3cm above/below the headtube/topdowntube intersection points.
        toprange = range(int(self.tubes['ht'].pt1[1])-CM2PX(0.2),int(self.tubes['ht'].pt1[1])-CM2PX(6.3),-1)
        # map search range from pixel units to the interpolated 0.1 pixel scale. note reversal here to search
        # in a negative direction.
        toprangeint = range(np.where(bxint==toprange[0])[0][0],np.where(bxint==toprange[-1])[0][0],-1)
        # BAYVIEW reduced botrange from 9.3 to 9 because it was overranging lrange defined above.
        # botrange = range(int(self.tubes['ht'].pt2[1])+CM2PX(0.2),int(self.tubes['ht'].pt2[1]+CM2PX(9.0)))
        # DYNAMITE_24 - reduced botrange again because of range error
        # hotrock - increased again slightly.
        # mxtrail - increased again slightly.
        # signal - long head tube. increased again
        botrange = range(int(self.tubes['ht'].pt2[1])+CM2PX(0.2),int(self.tubes['ht'].pt2[1]+CM2PX(8.0)))
        botrangeint = range(np.where(bxint==botrange[0])[0][0],np.where(bxint==botrange[-1])[0][0])
        # average the three colour bands
        mbspl = np.mean(np.abs(bspl),axis=1)

        plt.figure(figNo)
        plt.subplot(2,1,1)
        plt.title('extendHeadTube')
        plt.plot(lrange,htprofile)
        plt.subplot(2,1,2)
        plt.plot(bxint,np.mean(np.abs(bspl),axis=1))
        plt.plot(bxint[botrangeint],mbspl[botrangeint],'r')
        plt.plot(bxint[toprangeint],mbspl[toprangeint],'r')
        # for top peak take 0th index the first peak in the derivative. 
        # for bottom peak, take 2nd index 3rd peak, allowing two peaks for the cable housing
        # should be able to ignore 1 with min_dist but didn't work? or more smoothing in the splines.
        # note these thresholds 0.4 are still hard-coded
        # and will be too high for any bikes that are black
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.4,min_dist=CM2PX(1))[2]+botrangeint[0] 
        # trail. take 4th peak past cable
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.4,min_dist=CM2PX(1))[3]+botrangeint[0]         
        #  xtcsljr - fix scaling in the min_dist arg. exclude 2nd peak of the cable with min_dist. won't work if cable 
        # is closer to bottom of the head tube than cable thicknesss (2mm)
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.4,min_dist=CM2PX(.4)*10)[1]+botrangeint[0]         
        # zulu - cable at 45 deg need wider min_dist
        botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.4,min_dist=CM2PX(.5)*10)[1]+botrangeint[0]         
        # charger - reduced threshold
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.2,min_dist=CM2PX(1))[2]+botrangeint[0] 
        # works. extra peaks for double cables.
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.2,min_dist=CM2PX(1))[4]+botrangeint[0] 
        # exceed,riprock, yamajama - reduced threshold and took first peak
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.12,min_dist=CM2PX(1))[0]+botrangeint[0] 
        # signal. need higher threshold for ruffles but no cables
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.2,min_dist=CM2PX(1))[0]+botrangeint[0] 
        # mxtrail, kato. lower threshold but 1st peak
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.12,min_dist=CM2PX(1))[0]+botrangeint[0] 
        # ewoc - cable created 3 peaks. more blur.
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.2,min_dist=CM2PX(1))[3]+botrangeint[0]
        # toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.4,min_dist=CM2PX(3))[0]
        # BAYVIEW - reduced threshold
        # toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.3,min_dist=CM2PX(3))[0]
        # zulu - reduce threshold. 
        toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.1,min_dist=CM2PX(3))[0]
        # mxtrail. all black. reduce threshold.
        # toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.07,min_dist=CM2PX(3))[0]
        # kato . cable affects top peak
        # toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.3,min_dist=CM2PX(3))[1]

        # adjust peak from the peak to the ell (5% threshold) in the spline derivative for more accuracy
        # in measureHeadTube
        # trail 10%. yamajama 5%. signal some ruffles 10%
        peaks = [toppeak,botpeak]
        for i,p in enumerate(peaks):
            while mbspl[p] > mbspl[peaks[i]]*0.2:
                p += pow(-1,i)
            peaks[i] = p
        toppeak,botpeak = peaks

        plt.plot(bxint[toppeak],mbspl[toppeak],'r+')
        plt.plot(bxint[botpeak],mbspl[botpeak],'r+')
        plt.show(block= not __debug__)
        figNo = figNo + 1
    
        headtubetoplength = self.tubes['ht'].pt1[1] - bxint[toppeak]
        headtubebotlength = bxint[botpeak] - self.tubes['ht'].pt2[1]
        self.tubes['ht'].pt2 += np.array([np.cos(self.tubes['ht'].A)*headtubebotlength,np.sin(self.tubes['ht'].A)*headtubebotlength])
        self.tubes['ht'].pt1 -= np.array([np.cos(self.tubes['ht'].A)*headtubetoplength,np.sin(self.tubes['ht'].A)*headtubetoplength])

    def addStays(self,rearhub):
        # need to check order of wheels
        self.tubes['cs'].pt1 = np.array(rearhub)
        self.tubes['cs'].pt2 = self.tubes['st'].pt2
        self.tubes['ss'].pt1 = self.tubes['cs'].pt1
        self.tubes['ss'].pt2 = self.tubes['st'].pt1

    def plotTubes(self,aimg,linew=2):
        # plot raw lines detection
        for tube in ['ht','st','dt','tt']:
            # for some reason can't follow this syntax with the line array as did with tuples
            #for x1,y1,x2,y2 in np.nditer(Lline):
            #    cv2.line(aimg2,(x1,y1),(x2,y2),(0,0,255),2)
            pt1 = np.array([0,self.tubes[tube].y(0)])
            pt2 = np.array([cols,self.tubes[tube].y(cols)])
            # cv2.line(aimg,tuple(self.tubes[tube].pt1.astype(int)),tuple(self.tubes[tube].pt2.astype(int)),(255,0,0),linew)
            cv2.line(aimg,tuple(pt1.astype(int)),tuple(pt2.astype(int)),(255,0,0),linew)
        plotFig(aimg)

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
   

def pts2eq(((x1,y1),(x2,y2))):
    if x1<>x2:
        m = float(y2-y1)/float(x2-x1)
        b = y1 - m * x1
    else:
        m = float('NaN')
        b = float('NaN')
    return [m,b]

# this should probably return array not list
def eq2pts((m,b),(x1,x2)):
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

def plotLines(bw,lines,blockFlag=False,cmap=None,title=None, figNum=None):
    # plot raw lines detection
    for line in lines:
        for x1,y1,x2,y2 in line:
            if cmap == "gray":
                cv2.line(bw,(x1,y1),(x2,y2),140,CM2PX(0.1))
            else:
                cv2.line(bw,(x1,y1),(x2,y2),(0,0,255),CM2PX(0.1))
    plotFig(bw,blockFlag,cmap=cmap,title=title, figNum=figNum)

def coord2angle(line):
    for x1,y1,x2,y2 in line:
        if x1<>x2:
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

def runFull(filename,mmpx):
    P = profilePhoto(filename,mmpx=mmpx)
    G = Geometry()
    wheels,chainring = P.houghCircles(P.imGRAY)
    # note sort. back wheel  returned first
    # decided not to use outer radius in hub estimation becuase tread pattern sometimes throws it off
    G.rw = Tire(np.mean(wheels[0:4:4],axis=0)[0:2],wheels[0,2],wheels[2,2])
    G.fw = Tire(np.mean(wheels[1:5:4],axis=0)[0:2],wheels[1,2],wheels[3,2])
    G.cr = Circle(chainring[0:2],chainring[2])

    # create working image with wheels masked out to unclutter for line detect
    P.imW = np.copy(P.imGRAY)
    P.imW = cv2.blur(P.imW,(5,5))
    # [0]have an extra dimension to get rid of here... 
    # P.maskCircle(np.concatenate((wheels,G.chainring),axis=0),P.imW)
    # P.maskCircle(np.reshape(np.append(G.rw.centre,G.rw.rOuter),(1,3)),P.imW)
    # P.maskCircle(np.reshape(np.append(G.cr.centre,G.cr.R),(1,3)),P.imW)
    # mantra. increase these radii
    P.maskCircle(np.reshape(np.append(G.fw.centre,G.rw.rOuter*1.1),(1,3)),P.imW)
    P.maskCircle(np.reshape(np.append(G.rw.centre,G.rw.rOuter*1.1),(1,3)),P.imW)
    P.maskCircle(np.reshape(np.append(G.cr.centre,G.cr.R*1.1),(1,3)),P.imW)
    # try to save more of the seatpost here 
    P.maskRect([(G.rw.centre[0],0,G.cr.centre[0],G.rw.centre[1]-G.rw.rOuter)],P.imW)

    # try preliminary thresh to eliminate paint graphic effects
    ret,P.imW = cv2.threshold(P.imW,240,255,cv2.THRESH_BINARY)
    G.T = Tubeset()
    # start with main tubes. old method.
    # avglines,meqs = P.houghLines(P.imW,P.imRGB,minlength=8.5)
    # frog62, cube240. increase min length, one way to avoid the chain confusing the top tube
    # should generally be a correct strategy although it fails for riprock with such a bent top tube.
    # another round of refinement on angles better. or should just improve the maskCircle. 
    avglines,meqs = P.houghLines(P.imW,P.imRGB,minlength=11.0)    
    G.T.assignTubeLines(avglines,meqs,['tt','dt'])

    # todo: create ROI with seat tube only
    # find seat tube. new method
    lines = P.houghLinesS(P.imW,minlength=10)
    P.imW = np.copy(P.imRGB)
    plotLines(P.imW,lines,False,title="seat tube line detection")
    G.T.createSeatTubeTarget(G.cr)
    G.T.assignSeatTubeLine(lines)

    # todo:  head tube ROI. mask out everything to the left of the tt/dt intersection
    # find head tube
    # metaHT. increase minlength 4 to 7, separate edge process (maybe not needed though)
    # try preliminary thresh to eliminate paint graphic effects
    # pineridge. increase threshold due to white lettering
    ret,P.imW = cv2.threshold(P.imW,245,255,cv2.THRESH_BINARY)
    lines = P.houghLinesS(P.imW,minlength=7,edgeprocess='head')
    P.imW = np.copy(P.imRGB)
    plotLines(P.imW,lines,False,title="head tube line detection")
    G.T.createHeadTubeTarget(G.fw,type='susp')
    G.T.assignHeadTubeLine(lines)

    P.imW=np.copy(P.imRGB)
    G.T.plotTubes(P.imW)
    plt.show(block= not __debug__)

    # set the tubes lengths according to intersections
    G.T.calcTubes()

    # process the fork for refinement of head tube angle. 
    P.imW = np.copy(P.imGRAY)
    lines,meqs = P.findForkLines(G.fw,minlength=5)
    G.T.modifyTubeLines(meqs,'ht',op='mean')
    # Creig-24. charger. mxxc. head tube estimate not good enough use fork only
    # G.T.modifyTubeLines(meqs,'ht',op='replace')

    # recalc after modfication 
    G.T.calcTubes()

    P.imW=np.copy(P.imRGB)
    G.T.plotTubes(P.imW)

    # find the length of the head tube 
    P.imW = np.copy(P.imRGB)
    # Creig-24. slight error because brake cable is below the bottom of tube. need an extra check on tube profile perpendicular
    G.T.extendHeadTube(P.imW)
    # add fork, chainstay, seatstay
    G.T.addStays(G.rw.centre)
    G.fork.pt1 = G.T.tubes['ht'].pt2
    G.fork.pt2 = G.fw.centre

    P.imW=np.copy(P.imRGB)
    P.imW = G.plotTubes(P.imW,G.T)
    plotFig(P.imW,False,title="head extend")

    # with head tube approximately correct, redo the head angle estimate with better measurement.
    # P.imW = np.copy(P.imRGB)
    # edge. color should be less needed for this profile defined by white background
    P.imW = np.copy(P.imGRAY)
    # mxtrail. try thresholding for this measurement. 
    # vertex. a small white reflection of black paint requires this threshold
    ret,P.imW = cv2.threshold(P.imW,240,255,cv2.THRESH_BINARY)
    meq = G.T.measureHeadTube(P.imW)
    if meq is not None:       
        G.T.modifyTubeLines(meq,'ht',op='mean')
        # kato. poor initial and very good secondary detection. a better combination might be averaging the slopes
        # whille allowing the centreline to be entirely goverened by the secondary detection
        # G.T.modifyTubeLines(meq,'ht',op='replace')

    # replot the tubelines
    P.imW=np.copy(P.imRGB)
    G.T.plotTubes(P.imW)

    # recalc the tubes
    G.T.calcTubes()
    P.imW = np.copy(P.imRGB)
    # reuse of extendHeadTube ought to work but might hit welding bumps
    G.T.extendHeadTube(P.imW)
    
    # redo fork
    G.fork.pt1 = G.T.tubes['ht'].pt2
    G.fork.axle2crown = G.fork.l()
    G.fork.calcOffset(G.T.tubes['ht'])

    # create output
    G.calcParams()
    G.printParams()

    # final block with blocking
    P.imW = np.copy(P.imRGB)
    P.imw = G.plotTubes(P.imW,G.T)
    plotFig(P.imw,True)


if __name__=='__main__':
    filename = sys.argv[1]
    mmpx = float(sys.argv[2])
    runFull(filename,mmpx)

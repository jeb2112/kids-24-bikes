from operator import eq
from ssl import PROTOCOL_TLSv1_1
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
from .misc import *
from geo.tube import Tube
from geo.profile import Profile,ProfileException
from geo.display import Display
# from geo.geometry import Convert


class Photo():
    def __init__(self,filename,mmpx=None):
        self.rows = 0
        self.cols = 0
        if mmpx is not None:
            self.mmpx = mmpx # mm per pixel
            self.cv = Convert(mmpx=mmpx)
            self.D = Display(mmpx=mmpx)
        self.filename = filename
        self.imRGB = None
        self.imGRAY = None
        self.imEDGE = None
        self.imANNO = None
        self.imW = None
        self.bg = 255
        self.loadImage(self.filename)

    def loadImage(self,filename):
        # global rows,cols
        im = imageio.imread(filename)
        # if png/alpha, remove alpha
        self.rows,self.cols = np.shape(im)[0:2]
        if np.shape(im)[2]==4:
            im = cv2.cvtColor(im,cv2.COLOR_BGRA2BGR)
        # may need this reverse?
        # else:
        #     im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        self.imRGB = im 
        self.imANNO = self.imRGB
        self.imGRAY = cv2.cvtColor(self.imRGB,cv2.COLOR_BGR2GRAY)
        self.bg = self.imGRAY[0,0]

class ProfilePhoto(Photo):
    # global rows,cols
    def __init__(self,filename,mmpx=None):
        super(ProfilePhoto,self).__init__(filename,mmpx=mmpx)

    def houghCircles(self,bw):
        # this preblur maybe helps get rid of the apparent line in the middle of a tube
        # due to the reflection of light but this hasn't been investigated much yet
        # cleary,cujo24,AceLTD up to charger
        # bw1 = cv2.blur(bw,(5,5))
        # AceLTD. merely converting the hard-coded 5x5 kernel from cleary to same equivalent size broke it
        # reverting to 5x5, still had more difficulty with this image and cujo or cleary for the circles
        # fluid. switch to separate blurring for wheel/chainring. better tread outer detection this way
        bw1 = cv2.blur(bw,(self.cv.CM2PX(1.6),self.cv.CM2PX(1.6)))
        bw2 = cv2.blur(bw,(5,5))
        bw3 = np.copy(bw)
        # try simply binarizing the image? seemed like a good idea but param2 had to get even smaller to detect
        # anything, and the detections were anywhere but the correct spot. maybe need to reblur again?
        # ret,bw = cv2.threshold(bw,240,255,cv2.THRESH_BINARY)
        # self.D.plotfig(bw3,False,cmap="gray",fignum=1)
        # plt.show()        
        # not sure about final param1,2 choices yet
        # cleary,cujo24
        # wheelsInner = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1,minDist=self.cv.CM2PX(60),param1=self.cv.CM2PX(22),param2=self.cv.CM2PX(16),minRadius=self.cv.CM2PX(13),maxRadius=self.cv.CM2PX(30))
        # AceLTD...Alite-24
        # wheelsInner = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.2,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(10),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(20),maxRadius=self.cv.CM2PX(30))        
        # Creig-24. had to downsample image out of memory, have to scale mmpx accordingly though 0.803 now
        # wheelsInner = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.2,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(10),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(20),maxRadius=self.cv.CM2PX(30))        
        # DYNAMITE_24. riprock
        # wheelsInner = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.2,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(8),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(20),maxRadius=self.cv.CM2PX(30))        
        # exceed. ewoc
        # wheelsInner = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.2,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(4),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(20),maxRadius=self.cv.CM2PX(30))        
        # frog62. never did pick up  the front inner correctly.
        # wheelsInner = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.1,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(3),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(20),maxRadius=self.cv.CM2PX(30))        
        # zulu. not picking up very well
        wheelsInner = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.1,minDist=self.cv.CM2PX(86),param1=self.cv.CM2PX(2),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(20),maxRadius=self.cv.CM2PX(28))        
        # mantra. further reduction in maxRadius
        # wheelsInner = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.2,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(8),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(20),maxRadius=self.cv.CM2PX(26))        
        # pineridge. a scaling factor error was confusing the fit here, may not need unique params
        # wheelsInner = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.1,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(4),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(20),maxRadius=self.cv.CM2PX(24))        
        # place rear wheel first in list. note extra 1st dimension in output of HoughCircles
        wheelsInner = wheelsInner[0,wheelsInner[0,:,0].argsort(),:]
        # cleary.png
        # wheelsOuter = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1,minDist=self.cv.CM2PX(60),param1=self.cv.CM2PX(16),param2=self.cv.CM2PX(13),minRadius=self.cv.CM2PX(26),maxRadius=self.cv.CM2PX(60))
        # cujo24.png
        # wheelsOuter = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1.2,minDist=self.cv.CM2PX(60),param1=self.cv.CM2PX(10),param2=self.cv.CM2PX(6),minRadius=self.cv.CM2PX(26),maxRadius=self.cv.CM2PX(36))
        # AceLTD 1.2 seemed to make a big difference compared to 1.0? or was it dropping param1 way down to 4
        # wheelsOuter = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1.2,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(4),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(26),maxRadius=self.cv.CM2PX(40))
        # Bayview. further drop of param1 down to 3 required. that fixed the outerwheels, but lost the headtube!
        # wheelsOuter = cv2.HoughCircles(bw1,cv2.HOUGH_GRADIENT,1.2,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(3),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(26),maxRadius=self.cv.CM2PX(40))
        # pineridge. 
        # wheelsOuter = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.1,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(4),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(28),maxRadius=self.cv.CM2PX(40))
        # signal. not quite there.
        wheelsOuter = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1.2,minDist=self.cv.CM2PX(90),param1=self.cv.CM2PX(3),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(28),maxRadius=self.cv.CM2PX(40))
        wheelsOuter = wheelsOuter[0,wheelsOuter[0,:,0].argsort(),:]
        # argsort indexing removed dummy 1st dimension 
        wheels = np.concatenate((wheelsInner,wheelsOuter),axis=0)
        # cleary,cujo24
        # chainring = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,
        #     1,minDist=self.cv.CM2PX(60),param1=self.cv.CM2PX(22),param2=self.cv.CM2PX(16),minRadius=self.cv.CM2PX(3),maxRadius=self.cv.CM2PX(10))[0]
        # AceLTD
        # chainring = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1,minDist=self.cv.CM2PX(60),param1=self.cv.CM2PX(10),param2=self.cv.CM2PX(6),minRadius=self.cv.CM2PX(3),maxRadius=self.cv.CM2PX(10))[0]
        # alite-24. this reduced minDist detects couple dozen, to pick up the chainring.
        # chainring = cv2.HoughCircles(bw,cv2.HOUGH_GRADIENT,1,minDist=self.cv.CM2PX(20),param1=self.cv.CM2PX(4),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(3),maxRadius=self.cv.CM2PX(10))[0]
        # fluid - didn't pick up the chairing or outer diameter properly
        # mantra - chainring detection with these params was skewed about 1cm high
        # chainring = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1,minDist=self.cv.CM2PX(20),param1=self.cv.CM2PX(3),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(3),maxRadius=self.cv.CM2PX(8))[0]
        # pineridge. line. have to reduce maxradius because of false selection in mid-air above bottom bracket. could also mask that region out better to 
        # retain the large radius.
        chainring = cv2.HoughCircles(bw2,cv2.HOUGH_GRADIENT,1,minDist=self.cv.CM2PX(20),param1=self.cv.CM2PX(3),param2=self.cv.CM2PX(2),minRadius=self.cv.CM2PX(3),maxRadius=self.cv.CM2PX(6))[0]
        # BAYVIEW. use wheel hubs to select chainring circle of more than 1 detected
        if len(chainring[0])>1:
            wx,wy = np.mean(wheels[:,0:2],axis=0)
            chainring = np.reshape(chainring[(np.sqrt(pow(np.abs(chainring[:,0]-wx),2)+pow(np.abs(chainring[:,1]-wy),2))).argmin()],(1,3))
        for c in wheels:
            # cv2.circle(bw3,(c[0],c[1]),int(c[2]),140,5)
            cv2.circle(bw3,(int(c[0]),int(c[1])),int(c[2]),140,5)
        wc = np.mean([wheelsInner[:,0:2],wheelsOuter[:,0:2]],axis=0)   
        for w in wheels:
            # cv2.circle(bw3,(w[0],w[1]),self.cv.CM2PX(0.1),140,-1)
            cv2.circle(bw3,(int(w[0]),int(w[1])),self.cv.CM2PX(0.1),140,-1)
        for c in chainring:
            cv2.circle(bw3,(int(c[0]),int(c[1])),int(c[2]),140,5)
        self.D.plotfig(bw3,False,cmap="gray",title='houghCircle: wheel detection')
        plt.show(block =  not __debug__)        
        
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
            cv2.circle(target,(int(c[0]),int(c[1])),int(c[2]+self.cv.CM2PX(0)),255,-1)        

    def maskRect(self,masks,target):
        # masks. list of rects defined by top left point, bottom right
        # target. target image
        for c in masks:
            cv2.rectangle(target,(int(c[0]),int(c[1])),(int(c[2]),int(c[3])),255,-1)        
        
    def selectCircle(self,maskcentre,maskradius,target):
        t2 = np.copy(target)
        cv2.circle(target,(int(maskcentre[0]),int(maskcentre[1])),int(maskradius+self.cv.CM2PX(0)),255,-1)        
        target = t2 - target

    def resetAnnotatedIm(self):
        self.imANNO = self.imRGB

    # single lines detection. this is probably what houghLines should be. line selection and combination
    # should be in a separate function
    def houghLinesS(self,bw,edgeprocess='bike',minlength=8.5):
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
        self.D.plotfig(edges,False,title="houghLinesS")
        plt.show(block= not __debug__)
        # self.D.plotfigCV(edges)
        # plt.show(block=__debug__)

        lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=self.cv.CM2PX(minlength))

        # todo. arrange all lines with pt1[0]<pt2[0]

        return(lines)


    # main lines detection
    def houghLines(self,bw,aw,houghProcess='p',edgeprocess='bike',minlength=8.5):
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
        self.D.plotfig(edges,False,title='houghLines: edges')
        # self.D.plotfigCV(edges)
        plt.show(block=  not __debug__)

        # line processing
        if houghProcess=='p':
        # probabilistic works better for short segments like head tube.
        # 1. estimate of headset might be better by clipping the rigid fork exacttly at the headtube. or for suspension,
        # using the fork too.
        # 2. average line for the downtube in biased for the cleary example, can't see why the averaing doesn't worok
            # cleary lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=self.cv.CM2PX(15.75))
            # AceLTD-Alite-24
            # lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=self.cv.CM2PX(8.5))
            # Creig-24 seem to miss the headtube so try 6.5. and stxdt at bottom bracket could be constrained by chainring
            lines = cv2.HoughLinesP(edges,rho=1.0,theta=np.pi/180,threshold=30,maxLineGap=20,minLineLength=self.cv.CM2PX(minlength))
            if (lines is not None):
                # plot raw lines detection
                bw2 = np.copy(aw)
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(bw2,(x1,y1),(x2,y2),(0,0,255),self.cv.CM2PX(0.2))
                self.D.plotfig(bw2,False,title='houghLines: raw lines')
                plt.show(block= not __debug__)
            else:
                print('No edges detected')

            # average matching pairs of lines. slope/intercept might not be ideal to select the matching pairs
            # becuase non-linear.
            eqns = np.zeros((np.shape(lines)[0],2))
            rhotheta = np.zeros((np.shape(lines)[0],2))
            for i,line in enumerate(lines):
                eqns[i,:] = pts2eq(line[0,0:2],line[0,2:4])
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
                # self.D.plotfig(bw2)
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
                        p = Profile(int(midpt[0]),int(midpt[1]),self.cv.CM2PX(5),self.cv.CM2PX(5),RAD2DEG(t.A)-90,self.imW,mmpx=self.mmpx)
                        p.setprofile()
                        try:
                            p.setpeaks1()
                            p.setwidth()
                            p.setedge()
                            p.centre = p.prof2pix((p.tx2,midpt[1]))[:,0]
                            # adjust rho according to the 2nd line of the pair
                            # keep track of sign here
                            t.setrhotheta(t.rho-(p.centre[0]-p.edge[0]),t.theta)
                            meq = np.array([t.m,t.b])
                        except ProfileException:
                            print("Peaks not detected correctly")
                            meq = np.array([0,0])
                            
                    # not using this logic for the fork detection yet
                    elif edgeprocess=='fork':
                        meq = meq1

                if meqs is None:
                    meqs = meq
                    avglines = eq2pts(meq,(0,self.cols))
                else:
                    meqs = np.append(meqs,meq,axis=0)
                    avglines = np.append(avglines,eq2pts(meq,(0,self.cols)),axis=0)
                # throw out only the used offsets 
                eqns = np.delete(eqns,np.concatenate((eqnset1a,eqnset1b),axis=0),axis=0)
                rhotheta = np.delete(rhotheta,np.concatenate((eqnset1a,eqnset1b),axis=0),axis=0)
                lines = np.delete(lines,np.concatenate((eqnset1a,eqnset1b),axis=0),axis=0)

            avglines = np.reshape(avglines,(len(meqs)//2,4))
            meqs = np.reshape(meqs,(len(meqs)//2,2))
        
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

        self.D.plotfig(imW,cmap="gray")
        plt.show(block= not __debug__)
        # up to Creig-24. charger
        # avglines,meqs = P.houghLines(imW,P.imRGB,edgeprocess='fork',minlength=7.5)
        # fluid had to reduce minlength 
        avglines,meqs = self.houghLines(imW,self.imRGB,edgeprocess='fork',minlength=minlength)
        return(avglines,meqs)


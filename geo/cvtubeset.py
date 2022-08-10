from operator import eq
from ssl import PROTOCOL_TLSv1_1
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import peakutils
import copy
# relative intra-package import. can use .misc for module in the same directory as this file,
# or geo.misc as well, resolves to the same thing.
from .misc import *
from geo.profile import Profile,ProfileException
from geo.display import Display
from geo.tube import Tubeset,Tube

#################
# class CVTubeset
#################
# methods for CV processing a profile photo to get the geometry

class CVTubeset(Tubeset):
    def __init__(self,mmpx=None):
        super(CVTubeset,self).__init__(mmpx=mmpx)
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
        self.tubes['ht'].rho = self.tubes['ht'].rho - self.cv.CM2PX(1.0*2.54/2)
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
                if np.abs(t.rho - self.targets['st'].rho) > self.cv.CM2PX(5.5) or RAD2DEG(np.abs(t.theta - self.targets['st'].theta)) > 10:
                    break
                else:
                    set1 = np.concatenate((set1,np.reshape(np.array([t.rho,t.theta]),(1,2))),axis=0)
            # take mean angle for improved centreline approx
            meantheta = np.mean(set1[1:,1])
            # update the target. this allows to use rho to detect which edge is which below
            self.createSeatTubeTarget(cr=None,A=meantheta)
            # sort lines into edges for averaging
            set1 = set1[1:]
            # sethist,setbins = np.histogram(set1,range(int(np.amin(set1[:,0]))-self.cv.CM2PX(0.5),int(np.amin(set1[:,0]))+self.cv.CM2PX(5.5),self.cv.CM2PX(1)))
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
            travel = self.cv.CM2PX(6.5)
        elif type=='rigid':
            travel = 0
        crown = self.cv.CM2PX(4.5)
        length = self.cv.CM2PX(10)

        axle2crown = fw.rOuter + (travel + crown)
        self.targets['ht'].pt1 = (fw.centre[0]-(axle2crown+length) * np.cos(self.targets['ht'].A),
                                fw.centre[1]- (axle2crown+length)*np.sin(self.targets['ht'].A))
        self.targets['ht'].pt2 = (fw.centre[0]-(axle2crown) * np.cos(self.targets['ht'].A),
                                fw.centre[1]- (axle2crown)*np.sin(self.targets['ht'].A))

    # try to mask only the seat, not the seatpost. add seatpost detection to this
    def createSeatTubeTarget(self,cr,A=None):
        length = self.cv.CM2PX(12)
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
            # b2 = self.tubes[tube].pt1[1] - m2*self.tubes[tube].pt1[0]
            b2 = np.mean((self.tubes[tube].pt1[1],self.tubes[tube].pt2[1])) - m2*np.mean((self.tubes[tube].pt1[0],self.tubes[tube].pt2[0]))
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
        if self.tubes['tt'].pt2[1] - self.tubes['dt'].pt2[1] > -self.cv.CM2PX(1):
            # kludge: alter the down tube to parallel the top tube. for straight top tube
            kludge=1
            if kludge==1:
                xint = ( self.tubes['tt'].b-self.tubes['dt'].b ) / ( self.tubes['dt'].m - self.tubes['tt'].m )
                # trail. decreased this hard-coded offset from 6 bback to 4
                self.tubes['dt'].pt2[0]= xint - self.cv.CM2PX(4)
                self.tubes['dt'].pt2[1] = self.tubes['dt'].y(self.tubes['dt'].pt2[0])
                self.tubes['gt'] = Tube()
                # y intercept for gusset tube, using the point of truncation
                b = self.tubes['dt'].pt2[1] - self.tubes['tt'].m*self.tubes['dt'].pt2[0]
                self.tubes['gt'].seteqn(self.tubes['tt'].m,b)
                self.tubes['gt'].pt1 = np.array([xint-self.cv.CM2PX(4),self.tubes['gt'].y(xint-self.cv.CM2PX(4))])
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

    # adjust first estimate of head tube based on the right rube edge only. ie need to make sure it is not past the
    # centreline to the left, where the welds will interfere with the extendHeadTube method.
    # v. similar to measureHeadTube method but only a half profile. 
    def shiftHeadTube(self,img):
        global figNo
        p=[]
        hnew = np.zeros((2,2))
        for i1 in range(0,2):
            if i1==0:
                htx,hty = self.tubes['ht'].pt1
            else:
                htx,hty = self.tubes['ht'].pt2
            p.append(Profile(int(htx),int(hty),self.cv.CM2PX(0.0),self.cv.CM2PX(4.0),RAD2DEG(self.tubes['ht'].A)-90,img,mmpx=self.cv.mmpx))
            p[i1].setprofile()
            p[i1].setrpeak2()
            p[i1].setwidth()
            # rotate the point 1cm left of the edge back to pixel coords and assign as the new head tube point
            hnew[:,i1] = p[i1].prof2pix((htx+p[i1].width-self.cv.CM2PX(1.5),hty))[:,0]
        self.tubes['ht'].setpts(hnew[:,0],hnew[:,1])

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
            p.append(Profile(int(htx),int(hty),self.cv.CM2PX(5.5),self.cv.CM2PX(3.0),RAD2DEG(self.tubes['ht'].A)-90,img,mmpx=self.cv.mmpx))
            # rotate around the current estimate of the head tube point
            # M = cv2.getRotationMatrix2D((htx,hty),self.tubes['ht'].A*180/np.pi-90,1)
            # Minv = np.concatenate((np.linalg.inv(M[:,0:2]),np.reshape(-np.matmul(np.linalg.inv(M[:,0:2]),M[:,2]),(2,1))),axis=1)
            # rimg = cv2.warpAffine(img,M,(cols,rows))
            # needed 4 cm to the left for the wider headtube of riprock
            # range = range(int(htx)-self.cv.CM2PX(4),int(htx)+self.cv.CM2PX(3))
            # needed less than 3cm to right cube240
            # adjust again for mxxc 4.5cm to left
            # kato initial line very close to right tube edge. reduce range
            # can detect this properly based on teh 255 background
            # range = range(int(htx)-self.cv.CM2PX(4.5),int(htx)+self.cv.CM2PX(2))
            # yamajama - tapered head tube need more on the bottom. problem with top measure on the right from brakes exclude with 2.5 for now
            # zulu increase right-hand range slightly
            # debug only
            # rimg2=np.copy(rimg)
            # D.plotLines(rimg2,p[i1].rangeline.astype(int),False,cmap="gray")

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

                    if hwidthold == 0.0 or (p2[hi].width - hwidthold < -0.01*hwidthold):
                        hwidthold = p2[hi].width
                        hi += 1
                        hy += pow(-1,i1)
                    elif p2[hi].width - hwidthold > 0.01*hwidthold:
                        print("measureHeadTube: converged")
                        hi -= 1
                        raise ProfileException
                    # or a large jump to increased width also cause to end iteration
                    else:
                        print("measureHeadTube: converged")
                        raise ProfileException

                    # not likely more than 1-2 cm of extended head tube.   
                    if hi==self.cv.CM2PX(1):
                        print("measureHeadTube: ending iteration at 1 cm")
                        raise ProfileException

            except ProfileException:
                pass

            # find the edges at the converted width
            p2[hi].setedge()

            # if consistent width detected, rotate back to pixel coordinates. 
            p2[hi].centre = p2[hi].prof2pix((p2[hi].tx2,hty))[:,0]

            # save iteration result
            p[i1] =  copy.deepcopy(p2[hi])

        # plots
        plt.figure(figNo)
        for i1 in range(0,2):
            plt.subplot(2,1,i1+1)
            if i1==0:
                plt.title('measureHeadTube')
            plt.plot(p[i1].bxint,p[i1].mbspl)
            plt.plot(p[i1].range,p[i1].profile)
            plt.plot(p[i1].bxint[p[i1].peaks1[0:2].astype(int)],p[i1].mbspl[p[i1].peaks1[0:2].astype(int)],'r+')

        plt.show(block= not __debug__)
        figNo = figNo + 1

        # mxxc. revert to use of separate values to allow for the correction of the head tube angle
        # also check for positive error value. the bias to the right of the original head tube detection
        # should bias these values to negative. 
        h1shift = p[0].centre[0]-self.tubes['ht'].pt1[0]
        h2shift = p[1].centre[0]-self.tubes['ht'].pt2[0]
        if h1shift < self.cv.CM2PX(0.5) and h2shift < self.cv.CM2PX(0.5):
            meq[0,:] = pts2eq(p[0].centre,p[1].centre)
            # remove previous gusset tube if present
            if 'gt' in self.tubes.keys():
                self.tubes.pop('gt')
        else:
            print("measureHeadTube: no detection")
            meq=None
        return meq

    def extendHeadTube(self,img):
        global figNo
        rows,cols = np.shape(img)[0:2]
        # point of rotation. could be the average of the 1st pass head tube pts, if they are evenly placed
        # but some times, the downtube will curve at the join, making the lower head tube pt2
        # artificially high. meanwhile the top tube will likely never curve at the join.
        # therefore probably should use only the top point, and skew the search range accordingly.
        htx,hty = np.round(np.mean([self.tubes['ht'].pt1,self.tubes['ht'].pt2],axis=0)).astype(int)
        p = Profile(int(htx),int(hty),self.cv.CM2PX(10),self.cv.CM2PX(10),RAD2DEG(self.tubes['ht'].A)-90,img,type='vert',mmpx=self.cv.mmpx)
        # variations on profile needed.
        # signal. long head tube 
        # lrange = range(hty-self.cv.CM2PX(10),hty+self.cv.CM2PX(12))
        # AceLTD. the central profile overlapped some welds which broke the detection.
        # bias the profile to the right away from the seat/down gusset. should be fixed by shiftHeadTube
        # htprofile = rimg[lrange,htx+self.cv.CM2PX(0.2)]
        # mxtrail. all black. 1 profle alone disapperaed in shadow. try combining two or three
        # htprofile = (rimg[lrange,htx+self.cv.CM2PX(0.2)] + rimg[lrange,htx+self.cv.CM2PX(0.8)])/2     
        # mxxc. opposite problem. don't use a bias
   
        # create detection search ranges. 3cm above/below the headtube/topdowntube intersection points.
        toprange = range(int(self.tubes['ht'].pt1[1])-self.cv.CM2PX(0.2),int(self.tubes['ht'].pt1[1])-self.cv.CM2PX(6.3),-1)
        # map search range from pixel units to the interpolated 0.1 pixel scale. note reversal here to search
        # in a negative direction.
        toprangeint = range(np.where(p.bxint==toprange[0])[0][0],np.where(p.bxint==toprange[-1])[0][0],-1)
        # BAYVIEW reduced botrange from 9.3 to 9 because it was overranging lrange defined above.
        # botrange = range(int(self.tubes['ht'].pt2[1])+self.cv.CM2PX(0.2),int(self.tubes['ht'].pt2[1]+self.cv.CM2PX(9.0)))
        # DYNAMITE_24 - reduced botrange again because of range error
        # hotrock - increased again slightly.
        # mxtrail - increased again slightly.
        # signal - long head tube. increased again
        botrange = range(int(self.tubes['ht'].pt2[1])+self.cv.CM2PX(0.2),int(self.tubes['ht'].pt2[1]+self.cv.CM2PX(8.0)))
        botrangeint = range(np.where(p.bxint==botrange[0])[0][0],np.where(p.bxint==botrange[-1])[0][0])

        plt.figure(figNo)
        plt.subplot(2,1,1)
        plt.title('extendHeadTube')
        for i,color in enumerate(['red','green','blue'],start=1):
            plt.plot(p.range,p.profile[:,i-1],color=color)
        plt.plot(p.range,p.mprofile,'k')
        plt.subplot(2,1,2)
        plt.plot(p.bxint,np.mean(np.abs(p.bspl),axis=1))
        plt.plot(p.bxint[botrangeint],p.mbspl[botrangeint],'r')
        plt.plot(p.bxint[toprangeint],p.mbspl[toprangeint],'r')

        # need to redo this with some logic for the number of peaks detected within a certain distance
        # to try to establish whehter or not cable is present intelligently.
        # or that could be a parameter. or make use of black colour for a non-black bikes.

        # for top peak take 0th index the first peak in the derivative. 
        # for bottom peak, take 2nd index 3rd peak, allowing two peaks for the cable housing
        # should be able to ignore 1 with min_dist but didn't work? or more smoothing in the splines.
        # note these thresholds 0.4 are still hard-coded
        # and will be too high for any bikes that are black
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.4,min_dist=self.cv.CM2PX(1))[2]+botrangeint[0] 
        # trail. take 4th peak past cable
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.4,min_dist=self.cv.CM2PX(1))[3]+botrangeint[0]         
        #  xtcsljr - fix scaling in the min_dist arg. exclude 2nd peak of the cable with min_dist. won't work if cable 
        # is closer to bottom of the head tube than cable thicknesss (2mm)
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.4,min_dist=self.cv.CM2PX(.4)*10)[1]+botrangeint[0]         
        # zulu - cable at 45 deg need wider min_dist
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.4,min_dist=self.cv.CM2PX(.5)*10)[1]+botrangeint[0]    
        # alite. no cable, but bad reflections. last element in the peak list is needed for the 2 iterations!
        botpeak = peakutils.indexes(p.mbspl[botrangeint],thres=0.4,min_dist=self.cv.CM2PX(.5)*10)[-1]+botrangeint[0]    
        # charger - reduced threshold
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.2,min_dist=self.cv.CM2PX(1))[2]+botrangeint[0] 
        # works. extra peaks for double cables.
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.2,min_dist=self.cv.CM2PX(1))[4]+botrangeint[0] 
        # exceed,riprock, yamajama - reduced threshold and took first peak
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.12,min_dist=self.cv.CM2PX(1))[0]+botrangeint[0] 
        # signal. need higher threshold for ruffles but no cables
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.2,min_dist=self.cv.CM2PX(1))[0]+botrangeint[0] 
        # mxtrail, kato. lower threshold but 1st peak
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.12,min_dist=self.cv.CM2PX(1))[0]+botrangeint[0] 
        # ewoc - cable created 3 peaks. more blur.
        # botpeak = peakutils.indexes(mbspl[botrangeint],thres=0.2,min_dist=self.cv.CM2PX(1))[3]+botrangeint[0]
        # toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.4,min_dist=self.cv.CM2PX(3))[0]
        # BAYVIEW - reduced threshold
        # toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.3,min_dist=self.cv.CM2PX(3))[0]
        # zulu - reduce threshold. 
        toppeak = toprangeint[0] - peakutils.indexes(p.mbspl[toprangeint],thres=0.1,min_dist=self.cv.CM2PX(3))[0]
        # mxtrail. all black. reduce threshold.
        # toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.07,min_dist=self.cv.CM2PX(3))[0]
        # kato . cable affects top peak
        # toppeak = toprangeint[0] - peakutils.indexes(mbspl[toprangeint],thres=0.3,min_dist=self.cv.CM2PX(3))[1]

        # adjust peak from the peak to the ell (5% threshold) in the spline derivative for more accuracy
        # in measureHeadTube
        # trail 10%. yamajama 5%. signal some ruffles 10%
        peaks = [toppeak,botpeak]
        for i,pk in enumerate(peaks):
            while p.mbspl[pk] > p.mbspl[peaks[i]]*0.2:
                pk += pow(-1,i)
            peaks[i] = pk
        toppeak,botpeak = peaks

        plt.plot(p.bxint[toppeak],p.mbspl[toppeak],'r+')
        plt.plot(p.bxint[botpeak],p.mbspl[botpeak],'r+')
        plt.show(block= not __debug__)
        figNo = figNo + 1
    
        # rotate the measured length from the profile back to the pixel coordinates
        self.tubes['ht'].pt1 = p.prof2pix((htx,p.bxint[toppeak]))[:,0]
        self.tubes['ht'].pt2 = p.prof2pix((htx,p.bxint[botpeak]))[:,0]

        # try boxcar
        res = p.setboxcar()
        plt.figure(figNo)
        plt.step(p.range, p.box(p.range, *res.x), where='mid', label='diff-ev')
        plt.plot(p.range, p.mprofile, '.')
        plt.legend()
        plt.show(block = not __debug__)
        figNo = figNo +1

    def addStays(self,rearhub):
        # need to check order of wheels
        self.tubes['cs'].pt1 = np.array(rearhub)
        self.tubes['cs'].pt2 = self.tubes['st'].pt2
        self.tubes['ss'].pt1 = self.tubes['cs'].pt1
        self.tubes['ss'].pt2 = self.tubes['st'].pt1



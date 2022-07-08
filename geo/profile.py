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
from geo.misc import *


###############
# class Profile
###############

class ProfileException(Exception):
    pass

# class for processing line profiles in image. for now the line profiles are obtained by
# specifying the angle of rotation of the source image and horz/vert. the class instances are intended
# to be used in iterations to find optimal values but this isn't fully conceived yet.
class Profile():
    def __init__(self,hx,hy,hl,hr,A,img,type='horz'):
        # centre of rotation, angle, for source image to obtain profile
        # hl,hr are the pixel ranges to test on either side of the given point.
        # above/below for vertical, left/right for horz
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
        if type=='horz':
            self.range = range(self.hx-hl,self.hx+hr)
            self.rangeline = np.reshape(np.array([self.range[0],self.hy,self.range[-1],self.hy]),(1,1,4))
        elif type=='vert':
            self.range = range(self.hy-hl,self.hy+hr)
            self.rangeline = np.reshape(np.array([self.hx,self.range[0],self.hx,self.range[-1]]),(1,1,4))
        self.profile = np.zeros(len(self.range))
        self.mprofile = np.zeros(len(self.range))
        # self.rangeline = np.reshape(np.array([self.range[0],self.hy,self.range[-1],self.hy]),(1,1,4))
        # arbitrary factor of 10 for interpolation
        self.bxint = np.round(np.arange(self.range[0],self.range[-1],.1)*10)/10
        self.bspl = np.zeros((len(self.bxint),3))  
        self.mbspl = np.zeros(len(self.bxint))       

        self.peaks = np.zeros((2,2))
        # two types of detection will be used and kept track of separately
        self.peaks1 = np.zeros(4)
        self.peaks2 = np.zeros(4)
        self.width1 = 0
        self.width2 = 0
        self.width = 0
        self.edge = np.zeros((2,2))
        self.centre = np.zeros((2,2))
        self.tx2 = np.zeros(2)
        self.shift = np.zeros(2)

        self.setrimg()
        self.setprofile(type=type)

    # convert final width of the profile to a pair of edges and a centre point
    def setedge(self):
        if self.width1 and self.width2:
            if self.width1 < self.width2:
                self.edge = self.bxint[self.peaks1[0:2].astype(int)]
            else:
                self.edge = self.bxint[self.peaks2[0:2].astype(int)]
        elif self.width1:
            self.edge = self.bxint[self.peaks1[0:2].astype(int)]
        elif self.width2:
            self.edge = self.bxint[self.peaks2[0:2].astype(int)]
        # new centreline of the tube
        self.tx2 = np.mean(self.edge,axis=0)

    # compare widths determined by two methods and choose the resultant width
    # the width refers to a tube across which the profile is taken so a misnomer 
    # for the 'length' of the profile.
    # the process for establishing the tube width is expected to be iterative so 
    # raises an exception if failure.
    def setwidth(self):
        if self.width1 and self.width2:
            self.width = min(self.width1,self.width2)
        elif self.width1 or self.width2:
            self.width = max(self.width1,self.width2)
        else:
            print("measureHeadTube: no width detected")
            raise ProfileException

    # second method for establishing tube width based on two adjacent peaks in the derivative on either side of the search point
    # min_dist criterion is therefore not used.
    def setpeaks2(self):
        peaks2 = peakutils.indexes(self.mbspl,thres=0.2,min_dist=CM2PX(0)*10)
        # two max peaks should be the main edges if not the only ones selected. may need threshold image here though
        # riprock,pineridge fails at pt1 due to narrow gap and brake
        # edge. with threshold image, can rely on two innermost peaks as the criterion. didn't work for xtcsljr
        if len(peaks2) >= 2:
            idx = np.searchsorted(peaks2,len(self.bxint)/2)
            self.peaks2[0:2] = peaks2[idx-1:idx+1]
            self.width2 = self.bxint[self.peaks2[1].astype(int)]-self.bxint[self.peaks2[0].astype(int)]
    # right peak only. ie left end of profile is considered to be inside an area.
    def setrpeak2(self):
        peaks2 = peakutils.indexes(self.mbspl,thres=0.2,min_dist=CM2PX(0)*10)
        # 1st peak should be the main right edge. may need threshold image here though
        if len(peaks2) >= 1:
            self.peaks2[1] = peaks2[0]
            self.width2 = self.bxint[self.peaks2[1].astype(int)]-self.bxint[0]

    # first method for establishing tube width based on the spline derivative amplitude, edges should have the
    # largest derivative values, but min_dist criterion helps some borderline cases.
    def setpeaks1(self):
        peaks1 = peakutils.indexes(self.mbspl,thres=0.2,min_dist=CM2PX(3)*10)
        if len(peaks1)>=2:
            self.peaks1[0:2] = np.sort(peaks1[self.mbspl[peaks1].argsort()][::-1][0:2])
            self.width1 = self.bxint[self.peaks1[1].astype(int)]-self.bxint[self.peaks1[0].astype(int)]
        else:
            raise ProfileException

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
    def setprofile(self,type='horz'):
        if type=='horz':
            self.profile = self.rimg[self.hy,self.range]
        elif type=='vert':
            self.profile = self.rimg[self.range,self.hx]            
        if self.ndim==1:
            self.profile = np.reshape(self.profile,(len(self.profile),1))
            self.mprofile = self.profile
        elif self.ndim==3:
            self.mprofile = np.mean(self.profile,axis=1)

        for i in range(0,self.ndim):
            # arbitrary smoothing factor
            bsp = scipy.interpolate.splrep(self.range,self.profile[:,i],np.ones(len(self.range)),k=3,s=len(self.bxint))
            self.bspl[:,i] = scipy.interpolate.splev(self.bxint,bsp,der=1)
        # pineridge. need color to get the measurement due to black gear trigger and no white gap
        self.mbspl = np.mean(np.abs(self.bspl[:,0:self.ndim]),axis=1)

    # instead of using peaks in the derivative of the spline fit, try a boxcar fit
    def setboxcar(self):
        res = scipy.optimize.differential_evolution(lambda p: np.sum((self.box(self.range, *p) - self.mprofile)**2), [[0, 255], [self.range[0], self.range[-1]], [CM2PX(8), CM2PX(11)]])
        return(res)

    def box(self, x, *p):
        height, center, width = p
        return height*(center-width/2 < x)*(x < center+width/2)
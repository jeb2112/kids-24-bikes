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
from geo.photo import profilePhoto
from geo.misc import *
from geo.tube import Tube,Tubeset

figNo = 1
rows = 0 
cols = 0


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
    G.T.createHeadTubeTarget(G.fw,type='susp')
    G.T.assignHeadTubeLine(lines)
    # np.concatenate((lines,np.reshape(G.T.tubes['ht'].pt1+G.T.tubes['ht'].pt2,(1,1,4))),axis=0)
    plotLines(P.imW,lines,False,title="head tube line detection")
    plotLines(P.imW,np.reshape(G.T.tubes['ht'].pt1+G.T.tubes['ht'].pt2,(1,1,4)).astype(int),False,title="head tube line detection",color=(255,0,0))

    P.imW=np.copy(P.imRGB)
    G.T.plotTubes(P.imW)
    plt.show(block= not __debug__)

    # set the tubes lengths according to intersections
    G.T.calcTubes()

    # process the fork for refinement of head tube angle. 
    P.imW = np.copy(P.imGRAY)
    lines,meqs = P.findForkLines(G.fw,minlength=5)
    # G.T.modifyTubeLines(meqs,'ht',op='mean')
    # alite. Creig-24. charger. mxxc. head tube estimate not good enough use fork only
    G.T.modifyTubeLines(meqs,'ht',op='replace')

    # further adjust the current head tube estimate using the right edge of the head tube
    # this could be done with thresholded silhouette
    # this might replace the fork lines estimate?
    P.imW=np.copy(P.imGRAY)
    ret,P.imW = cv2.threshold(P.imW,245,255,cv2.THRESH_BINARY)
    G.T.shiftHeadTube(P.imW)

    # recalc after modfication 
    G.T.calcTubes()

    P.imW=np.copy(P.imRGB)
    G.T.plotTubes(P.imW)
    plt.show(block= not __debug__)

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

# for a text file list of images and mmpx scale factors
def readFile(filename):
    if filename.closed:
        f = open(filename,'r')
    else:
        f = filename
    flist=[]
    for line in f:
        flist.append([re.search('^[a-zA-Z\/]*\.(png|jpg)',line).group(0),float(re.search('[0|1]\.[0-9]*',line).group(0))])
    f.close()
    return(flist)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    # quick hack for 2.x type=file, to fix properly
    group.add_argument("-f","--file",type=argparse.FileType('r'),dest="file",help="filename containing list of photos to process")
    group.add_argument("-i","--img",type=argparse.FileType('r'),dest="img",help="filename of a photos to process")
    parser.add_argument("--mmpx",type=float,dest="mmpx",help="millimetres per pixel scaling")
    parser.add_argument("-p","--plot",dest="plot",help="option for plots")
    
    args =  parser.parse_args()
    
    if args.img:
        mmpx = args.mmpx
        runFull(args.img,args.mmpx)
    elif args.file:
        flist = readFile(args.file)
        for n,m in flist:
            mmpx = m
            runFull(n,m)
        

from operator import eq
from ssl import PROTOCOL_TLSv1_1
import numpy as np
import cv2
import matplotlib.pyplot as plt
from geo.misc import Convert

figNo = 1
# rows = 0 
# cols = 0

#################
# general methods
#################
class Display():
    def __init__(self,mmpx=None):
        if mmpx is not None:
            self.cv = Convert(mmpx=mmpx)

    def plotFig(self,img,blockflag=False,cmap=None,title=None,fignum=None):
        global figNo
        if fignum is None:
            fignum = figNo
            figNo += 1

        plt.figure(fignum)
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
        plt.show(block=blockflag)
        # figNo += 1

    # CV graphics do work with pause for keypad that way plt.ion() was supposed to
    def plotFigCV(self,img,title="Fig"):
        # this should work but doesn't
        # cv2.startWindowThread()
        cv2.namedWindow(title)
        cv2.imshow(title,img)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def plotLines(self,bw,lines,blockflag=False,cmap=None,title=None, fignum=None, color=(0,0,255)):
        # plot raw lines detection
        for line in lines:
            for x1,y1,x2,y2 in line:
                if cmap == "gray":
                    cv2.line(bw,(x1,y1),(x2,y2),140,self.cv.CM2PX(0.1))
                else:
                    cv2.line(bw,(x1,y1),(x2,y2),color,self.cv.CM2PX(0.1))
        self.plotFig(bw,blockflag,cmap=cmap,title=title, fignum=fignum)

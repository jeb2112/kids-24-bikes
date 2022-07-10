from operator import eq
from ssl import PROTOCOL_TLSv1_1
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from geo.photo import ProfilePhoto
from geo.misc import *
from geo.geometry import CVGeometry,AnnoGeometry


def runCV(filename,mmpx):
    P = ProfilePhoto(filename,mmpx=mmpx)
    G = CVGeometry(mmpx=mmpx)
    P = G.findwheels(P)
    P = G.findlines(P)


def runAnnotate(flist):
    for filename,mmpx in flist:
        P = ProfilePhoto(filename,mmpx=mmpx)
        G = AnnoGeometry(mmpx=mmpx)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    # quick hack for 2.x type=file, todo fix properly
    group.add_argument("-f","--file",type=argparse.FileType('r'),dest="file",help="filename containing list of photos to process")
    group.add_argument("-i","--img",type=argparse.FileType('r'),dest="img",help="filename of a photos to process")
    parser.add_argument("--mmpx",type=float,dest="mmpx",help="millimetres per pixel scaling")
    parser.add_argument("-p","--plot",dest="plot",help="option for plots")
    parser.add_argument("--annotate",action="store_true",default=False)
    parser.add_argument("--cv",action="store_true",default=True)
    
    args =  parser.parse_args()
    
    if args.annotate:
        if args.file:
            flist = readFile(args.file)
            runAnnotate(flist)
    elif args.cv:
        if args.img:
            mmpx = args.mmpx
            runCV(args.img,args.mmpx)
        if args.file:
            flist = readFile(args.file)
            for n,m in flist:
                mmpx = m
                runCV(n,m)
        

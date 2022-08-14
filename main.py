from operator import eq
from ssl import PROTOCOL_TLSv1_1
import os
import matplotlib.pyplot as plt
import argparse
from geo.photo import ProfilePhoto
from geo.misc import *
from geo.geometry import CVGeometry,AnnoGeometry
from scrape.gsheet import Gsheet
from scrape.scraper import Scraper
from process.process import Process
from process.profile import Profile

def runCV(filename,mmpx):
    P = ProfilePhoto(filename,mmpx=mmpx)
    name = os.path.splitext(os.path.basename(filename))[0]
    G = CVGeometry(name=name,mmpx=mmpx)
    P = G.findwheels(P)
    P = G.findlines(P)


def runAnnotate(flist):
    for filename,mmpx in flist:
        P = ProfilePhoto(filename,mmpx=mmpx)
        name = os.path.splitext(os.path.basename(filename))[0]
        G = AnnoGeometry(name=name,mmpx=mmpx)
        G.findwheels(P)
        G.findlines(P)
        G.calc()
        G.plot(P)
        G.save()
        a=1

def runScrape():
    gs = Gsheet(online=True)
    sc = Scraper()
    bcol = 'R' # starting column for debugging
    b1 = ord(bcol[-1])-64-4
    if len(bcol)==2:
        b1 += (ord(bcol[0])-64)*26
    for b in gs.bikes[b1:]:
        print('item# {}, {}, {}'.format(b1,b['label'],b['build']))
        sc.dosoup(b)
        b1 += 1

def runProcess():
    p = Process(pad=False)
    ret = p.removedupl()
    if ret:
        p.clearprocessed()
    p.runprocess()

def runProfile():
    p = Profile(kfold=4,modelname='profile_model_adam',domodel=True)
    p.dotrain()

def plotProfile():
    p = Profile(kfold=4,modelname='profile_model_adam',domodel=False)
    p.plot()

def predictProfile():
    p = Profile(kfold=4,modelname='profile_model_adam',domodel=True)
    p.dopredict()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    # quick hack for 2.x type=file, todo fix properly
    group.add_argument("-f","--file",type=argparse.FileType('r'),dest="file",help="filename containing list of photos to process")
    group.add_argument("-i","--img",type=argparse.FileType('r'),dest="img",help="filename of a photos to process")
    parser.add_argument("--mmpx",type=float,dest="mmpx",help="millimetres per pixel scaling")
    parser.add_argument("-p","--plot",dest="plot",help="option for plots")
    parser.add_argument("--annotate",action="store_true",default=False)
    parser.add_argument("--cv",action="store_true",default=False)
    parser.add_argument("--scrape",action="store_true",default=False)
    parser.add_argument("--process",action="store_true",default=False)
    parser.add_argument("--trainprofile",action="store_true",default=False)
    parser.add_argument("--plotprofile",action="store_true",default=False)
    parser.add_argument("--predictprofile",action="store_true",default=False)
    
    args =  parser.parse_args()
    
    if args.predictprofile:
        predictProfile()
    if args.plotprofile:
        plotProfile()
    if args.trainprofile:
        runProfile()
    if args.process:
        runProcess()
    if args.annotate:
        if args.file:
            flist = readFile(args.file)
            runAnnotate(flist)
    elif args.scrape:
        runScrape()
    elif args.cv:
        if args.img:
            mmpx = args.mmpx
            runCV(args.img,args.mmpx)
        if args.file:
            flist = readFile(args.file)
            for n,m in flist:
                mmpx = m
                runCV(n,m)
        

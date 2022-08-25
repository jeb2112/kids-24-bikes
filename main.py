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
from process.gtable import GTable
from process.gtableocr import GTableOCR
# from process.detector import Detector

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
    bcol = 'CC' # starting column for debugging
    b1 = ord(bcol[-1])-64-4
    if len(bcol)==2:
        b1 += (ord(bcol[0])-64)*26
    for b in gs.bikes[b1:]:
        print('item# {}, {}, {}'.format(b1,b['label'],b['build']))
        sc.dosoup(b)
        b1 += 1

# dataset prep
# profile images
def runProcess(rootname,clear=True):
    p = Process(pad=False,rootname=rootname,tx=320,ty=240)
    ret = p.removedupl()
    if clear:
        p.clearprocessed()
    p.runprocess()
# ocr
def runOCR(rootname,clear=True):
    p = Process(pad=False,rootname=rootname,processname='ocr')
    ret = p.removedupl()
    if clear:
        p.clearprocessed()
    p.runocr()

# training
# profile photos
def runProfile():
    p = Profile(kfold=4,modelname='profile_model_adam',domodel=True)
    p.dotrain()

# ocr
def trainOCR():
    p = GTableOCR(kfold=4)
    p.dotrain()


def plotModel(modelname):
    modelclasses = {'profile':Profile,'gtable':GTable}
    modelclass = modelname.split('_')[0]
    if modelclass in modelclasses.keys():
        p = modelclasses[modelclass](kfold=4,modelname=modelname,domodel=False)
        p.plot()
    else:
        return

def predictProfile():
    p = Profile(kfold=4,modelname='profile_model_adam',domodel=True)
    p.dopredict()

# geometry tables
def runGTable():
    g = GTable(kfold=4,modelname='gtable_model_adam',domodel=True)
    g.dotrain()


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
    parser.add_argument("--processocr",action="store_true",default=False)
    parser.add_argument("--trainprofile",action="store_true",default=False)
    parser.add_argument("--trainocr",action="store_true",default=False)
    parser.add_argument("--plotmodel",action="store_true",default=False)
    parser.add_argument("--predictprofile",action="store_true",default=False)
    parser.add_argument("--traingtable",action="store_true",default=False)
    parser.add_argument("--plotgtable",action="store_true",default=False)
    parser.add_argument("--predictgtable",action="store_true",default=False)
    parser.add_argument("--rootname",type=str,help='root filename for image preprocessing',default=None)
    parser.add_argument("--modelname",type=str,help='model filename for various',default=None)
    
    args =  parser.parse_args()
    
    if args.predictprofile:
        predictProfile()
    if args.plotmodel:
        if args.modelname is None:
            raise argparse.ArgumentError('No model name given')
        plotModel(args.modelname)
    if args.trainprofile:
        runProfile()
    if args.trainocr:
        trainOCR()

    if args.traingtable:
        runGTable()

    if args.process:
        if args.rootname is None:
            raise argparse.ArgumentError('No filename given')
        runProcess(args.rootname)
    if args.processocr:
        if args.rootname is None:
            raise argparse.ArgumentError('No filename given')
        runOCR(args.rootname)

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
        

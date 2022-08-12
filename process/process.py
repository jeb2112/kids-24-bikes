import os
from PIL import Image,ImageOps,ImageStat
import imghdr
import numpy as np
import re
import filecmp
import glob

# class for pre-processing training and test images
class Process():
    def __init__(self,pad=False):
        pngdir = '/home/src/kids-24-bikes/png'
        self.a = 'nonprofile'
        self.b = 'profile'
        self.adir = os.path.join(pngdir,self.a)
        self.bdir = os.path.join(pngdir,self.b)
        self.a_processed = 'nonprofile_processed'
        self.b_processed = 'profile_processed'
        self.adir_processed = os.path.join(pngdir,self.a_processed)
        self.bdir_processed = os.path.join(pngdir,self.b_processed)
        self.tx = 160
        self.ty = 120
        if not os.path.exists(self.adir_processed):
            os.mkdir(self.adir_processed)
        if not os.path.exists(self.bdir_processed):
            os.mkdir(self.bdir_processed)
        if pad:
            self.padfilename()

    # convenience method for one-time use zero-padding input filenames
    def padfilename(self):
        pattern = re.compile(r'(\d+)')
        for di in [self.adir,self.bdir]:
            flist = os.listdir(di)
            for f in flist:
                f0 = f[:]
                a = re.search(pattern,f)[0]
                a0 = a.zfill(3)
                f0 = f0.replace(a,a0)
                os.rename(os.path.join(di,f),os.path.join(di,f0))

    # occasional use, remove any duplicates
    def removedupl(self):
        ret = False
        for di in [self.adir,self.bdir]:
            flist = os.listdir(di)
            for i,f1 in enumerate(flist):
                for f2 in flist[i+1:]:
                    fpath1 = os.path.join(di,f1)
                    fpath2 = os.path.join(di,f2)
                    if filecmp.cmp(fpath1,fpath2):
                        os.remove(fpath1)
                        ret = True
                        break
        return ret

    def clearprocessed(self):
        for di in [self.adir_processed,self.bdir_processed]:
            files = glob.glob(di+'/*')
            for f in files:
                os.remove(f)

    # main method
    def runprocess(self):
        for (di,do) in zip([self.adir,self.bdir],[self.adir_processed,self.bdir_processed]):
            flist = os.listdir(di)
            flist.sort()
            for i,f in enumerate(flist):
                ipath = os.path.join(di,f)
                img_ext = '.'+imghdr.what(ipath)
                if img_ext != 'png':
                    pass
                img_name,img_ext2 = f.split('.')
                opath = os.path.join(do,img_name)
                if os.path.exists(opath):
                    continue
                im = Image.open(ipath)
                if any(ext in im.mode for ext in ['RGB','CMY','HSV','YCbCr']):
                    im = self.rgb2g(im)
                im = self.resize(im)
                im.save(opath,format='png')
                continue
        return


    # convert to grayscale
    def rgb2g(self,I):
        ir = ImageOps.grayscale(I)
        return ir

    # crude estimate background for padding
    def est_bg(self,I):
        stat = ImageStat.Stat(I)
        return stat.median

    # resize to targets sizes with padding
    def resize(self,I):
        bg = self.est_bg(I)
        Ir = np.ones((self.ty,self.tx))*bg
        w,h = I.size
        if w/h > self.tx/self.ty:
            tx = self.tx
            ty = int ((self.tx/self.ty)/(w/h) * self.ty)
            ty_offset = int( (self.ty-ty)/2)
            tx_offset = 0
        else:
            ty = self.ty
            tx = int((self.ty/self.tx)/(h/w) * self.tx)
            tx_offset = int( (self.tx-tx)/2)
            ty_offset = 0
        ir = I.resize((tx,ty),Image.LANCZOS)
        Ir[ty_offset:ty_offset+ty,tx_offset:tx_offset+tx] = ir
        Ip = Image.fromarray(Ir.astype(np.uint8),mode='L')
        return Ip
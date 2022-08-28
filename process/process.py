import os
from PIL import Image,ImageOps,ImageStat,ImageFont
from pytesseract import pytesseract
import imghdr
import numpy as np
import matplotlib.pyplot as plt
import re
import filecmp
import glob
import tempfile
import nltk
import warnings


# class for pre-processing training and test images
# currently coded for two classes
class Process():
    def __init__(self,pad=False,rootname='profile',processname='processed',tx=160,ty=120):
        pngdir = '/home/src/kids-24-bikes/png/traindata'
        self.a = 'non'+rootname
        self.b = rootname
        self.adir = os.path.join(pngdir,'raw',self.a)
        self.bdir = os.path.join(pngdir,'raw',self.b)
        self.a_processed = self.a+'_'+processname
        self.b_processed = self.b+'_'+processname
        self.adir_processed = os.path.join(pngdir,processname,self.a_processed)
        self.bdir_processed = os.path.join(pngdir,processname,self.b_processed)
        self.tx = tx
        self.ty = ty
        if not os.path.exists(self.adir_processed):
            os.makedirs(self.adir_processed)
        if not os.path.exists(self.bdir_processed):
            os.makedirs(self.bdir_processed)
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

    # main method for images, all input files
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
                im = self.remove_transparency(im)
                if any(ext in im.mode for ext in ['RGB','CMY','HSV','YCbCr']):
                    im = self.rgb2g(im)
                im = self.resize(im)
                im.save(opath,format='png')
                continue
        return

    # process a single img
    def runsingle(self,img):
        if 'PIL' not in str(type(img)):
            raise Exception('Expected PIL image')
        #check for transparency
        img = self.remove_transparency(img)
        img = self.rgb2g(img)
        img = self.resize(img)
        return img #PIL

    # main method for ocr
    def runocr(self,doplot=False,debugstart=None):
        pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        for (di,do) in zip([self.adir,self.bdir],[self.adir_processed,self.bdir_processed]):
            flist = os.listdir(di)
            flist.sort()
            for i,f in enumerate(flist):
                print(i)
                if debugstart is not None: # quick check for debugging
                    if i <= debugstart or di == self.adir:
                        continue
                ipath = os.path.join(di,f)
                img_name,img_ext = f.split('.')
                opath = os.path.join(do,img_name+'.txt')
                if os.path.exists(opath):
                    continue
                im = Image.open(ipath)
                imdpifile = self.set_imagedpi(im)
                im = Image.open(imdpifile)
                os.remove(imdpifile)
                # if any(ext in im.mode for ext in ['RGB','CMY','HSV','YCbCr']):
                #     im = self.rgb2g(im)
                im = self.remove_transparency(im)
                bg = self.est_bg(im)
                if np.mean(np.array(bg)) < 128: # tesseract only does dark text on light background
                    if im.mode != 'P':
                        im = ImageOps.invert(im)
                    elif im.mode == 'P':
                        im = im.convert('L')
                        im = ImageOps.invert(im)
                if True:
                    imscale=4
                    im = im.resize((imscale*im.size[0],imscale*im.size[1]),resample=Image.Resampling(1))
                tex = pytesseract.image_to_string(im,config='--psm 11')
                tex = self.preprocessocr(tex)
                if len(tex) == 0:
                    warnings.warn('Found no text: {}'.format(img_name))
                    continue
                with open(opath,'w') as fp:
                    tex = img_name + ' ' + tex
                    fp.write(tex)
                if doplot:
                    plt.figure(figsize=(12,4))
                    plt.subplot(1,2,1)
                    plt.imshow(im)
                    plt.title(img_name)
                    plt.subplot(1,2,2)
                    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
                            labelbottom=False,labelleft=False)
                    plt.box(False)
                    if di == self.adir:
                        font_size = 14
                    elif di == self.bdir:
                        font_size = 12
                    teststring =  ' '.join(tex.split('\n'))
                    teststring = self.wraptext(teststring,width=3,fontsize=font_size)
                    plt.text(0,0.0,teststring,wrap=False,fontsize=font_size)
                    plt.show()
                    a=1

    def preprocessocr(self,txt):
        shortwords = set(['m','l','s','cm','mm','bb','xs','xl','wb','tt','st','ht','so','cc','ct','c-c','c-t','cs','soh','sa','ha'])
        words = set(nltk.corpus.words.words())
        txt = txt.replace('\n',' ')
        txt = txt.replace('\\','')
        tkns = nltk.word_tokenize(txt)
        new_txt = ''
        for w in tkns:
            # in case a frame dimension is parsed in mm units with no blank, just want the number
            if re.fullmatch('[0-9]+mm',w,flags=re.IGNORECASE):
                w = w.replace('mm','')
            # round all numbers to single significant digit
            if re.fullmatch('[0-9]+',w):
                w = str(np.around(int(w),decimals=-(len(w)-1)))
            # general match
            if (w.lower() in words and len(w.lower()) == 3) or \
                w.lower() in shortwords or \
                re.fullmatch('[0-9]+',w) or \
                re.fullmatch('[a-z]{4,}',w,flags=re.IGNORECASE):
                new_txt += w.lower() + ' '
        return new_txt

    # convert transparency, def white
    def remove_transparency(self,img,bgcolor=(255,255,255)):
        if (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
        if img.mode in ('RGBA','LA'):
            alpha = img.getchannel('A')
            if 0 in np.array(alpha):
                bg = Image.new('RGBA',img.size,bgcolor+(255,))
                bg.paste(img,mask=alpha)
                return bg.convert('RGB')
            else:
                return img.convert('RGB')
        else:
            return img

    # convert to grayscale
    def rgb2g(self,I):
        ir = ImageOps.grayscale(I)
        return ir

    # crude estimate background for padding
    def est_bg(self,I):
        if I.mode == 'P':
            I = I.convert('L')
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

    # for ocr
    # convert to 300 dpi. only works via save to disk?
    def set_imagedpi(self,I,dpival=300):
        image_dpi = I
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_filename = temp_file.name
        image_dpi.save(temp_filename, dpi=(dpival, dpival))
        return temp_filename     

    # quick hack for printing wrapped text in a plt figure box
    def wraptext(self,teststring,width=3,fontsize=14):
        currentfont = 'DejaVuSans' # matplotlib default
        mydpi = 96 # screen resolution
        testfont = ImageFont.truetype(currentfont,fontsize)
        stringsize = np.array(testfont.getsize(teststring)) / mydpi
        rowsize = int(width / stringsize[0] * len(teststring))
        rindex=0
        while rindex+rowsize < len(teststring):
            ri = teststring.rfind(' ',0,rindex+rowsize)
            teststring = teststring[:ri] + '\n' + teststring[ri+1:]
            rindex += rowsize
        return teststring

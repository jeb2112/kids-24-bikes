import os
import io
from ssl import SSL_ERROR_EOF
from urllib3.exceptions import SSLError
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import random
import glob
import re
import pickle
import warnings
import requests
import imghdr
import numpy as np
from PIL import Image
from ast import literal_eval
from datetime import date
import time
from process.profile import Profile
from process.process import Process
from scrape.scraper import Scraper


class ImgScraper(Scraper):
    def __init__(self,**kwargs):
        super(ImgScraper,self).__init__(self,**kwargs)
        self.profiler = Profile(kfold=4,modelname='profile_model_adam',domodel=True)
        self.processor = Process(pad=False)
        return

    #############
    # main method
    #############

    def dosoup(self,b,debug=True):

        self.setpattern(b)
        # for dev: if already have the photo don't hammer the server
        # TODO: check for fork with a different trail
        if glob.glob(self.fname+'.*'):
            warnings.warn('Photo already exists, returning...')
            return

        ptext = self.getpage()
        if ptext is None:
            warnings.warn('No page text retrieved, skipping...')
            return
        soup = BeautifulSoup(ptext,'html.parser')
        imgs = soup.find_all('img')
        meta = soup.find_all('meta')
        links = soup.find_all('a')
        urls = []

        # form a short list of candidate images
        profileimg_url = None
        # start with full build pattern, then fall back to model pattern
        for p in [self.build_pattern,self.model_pattern]:

            if len(imgs) > 0:
                urls = self.get_imgurls(imgs,p)
            if len(meta) > 0:
                meta_urls = self.get_metaurls(meta,p)
                if meta_urls is not None:
                    urls = urls + meta_urls
            if len(urls) == 0:
                warnings.warn('No img or meta urls found. continuing')
                return

            # go through the list and take the largest profile image
            profileimg_url,ext = self.search_urls(urls,b)
            if profileimg_url:
                break

        if profileimg_url is None:
            warnings.warn('No profile image found, skipping...')
            return
        self.saveimage(profileimg_url,ext)


    #############
    # aux methods
    #############

    # process a list of urls and take largest profile image
    def search_urls(self,urls,b):
        maxsize = 0
        maximgurl = None
        for url in urls:
            url = self.process_url(url,b)
            if url is None:
                continue
            imgurl,ext = self.getimage(url,dosave=False)
            if imgurl is None:
                continue
            w,h = imgurl.size
            img = self.processor.runsingle(imgurl)
            img = np.array(img)/255. # norm
            img = np.reshape(img,(1,self.processor.ty,self.processor.tx,1))
            p0 = self.profiler.test(img)
            if p0:
                if w*h > maxsize:
                    maxsize,maximgurl = w*h,imgurl
                if maxsize > 1e6: # big enough, avoid hammering website
                    break
            else:
                a=1
                continue
        return maximgurl,ext


    # get a short list of candidate image urls
    def get_imgurls(self,imgs,p=None):
        urls = []
        if p == None:
            p = self.build_pattern

        for i in imgs:
            # most generally look for name tag in the src attribute
            if re.search(p,i.attrs.get('src',''),flags=re.I): # khs alite
                img_url = self.process_srcset(p,i)
                urls.append(img_url)
                continue
            
            # also try the data-src attribute
            # early seeker has src that is small, data-src with widths.
            if re.search(p,i.attrs.get('data-src',''),flags=re.I): # khs alite
                img_url = self.process_srcset(p,i,defkey='data-src')
                urls.append(img_url)
                continue

            # next try the alt attribute
            if re.search(p,i.attrs.get('alt',''),flags=re.I):
                # dback pines doesn't have name in the 'src' attr, but it can be 
                # inferred by a data-widths attr. meanwhile, lack of a lazyload
                # method blocks the correct assignment. so this is just a kludge
                # for now may or may not need to be permanent.
                if 'data-widths' in i.attrs.keys():
                    img_url = self.process_srcset(p,i)
                    urls.append(img_url)
                    continue
                # TODO handle variations on 'src'
                if 'src' in i.attrs.keys():
                    img_url = i.attrs['src']
                    urls.append(img_url)
                    continue

            # try the class attribute
            if False:
                if re.search(p,i.attrs.get('class',''),flags=re.I):
                    # not sure whether to use lazyload or not
                    # for MTB 69, it is needed to get the largest image
                    if 'lazyload' in i.attrs.get('class',''):
                        # TODO method to process a lazyload class
                        continue

            # other problem cases
            # scout 26 also broken
            # vpace max 24,26 same. getting only a small image. 
            # amulet 24,26 maybe another header tag needed? page response seems wrong-different than browser
            # roscoe profile image is by some lazyload class, don't pick up the <img>, 
            # in inspector something about an event listener, may need selenium

        return urls

    # search meta tags for list of possible images
    def get_metaurls(self,meta,p):
        urls = []
        for m in meta:
            if m.attrs.get('content','').startswith("http"): # for now assume we have this
                if re.search(p,m.attrs['content'],flags=re.I):
                    if any(ext in m.attrs['content'] for ext in ['gif','jpg','png','webp']):
                        img_url = m.attrs['content']
                        urls.append(img_url)
                        # return img_url
                else: # for opus recon. risky but take any .png in a meta content
                    if any(ext in m.attrs['content'] for ext in ['gif','jpg','png','webp']):
                        img_url = m.attrs['content']
                        urls.append(img_url)
                        # return img_url
        # no images found, repeat and take just a url. eg special riprock
        if not len(urls):
            for m in meta:
                if m.attrs.get('content','').startswith('http'):
                    if re.search(p,m.attrs['content'],flags=re.I):
                        img_url = m.attrs['content'] # for special riprock, no ext but
                                                        # can at least check build
                        urls.append(img_url) 

        return urls

    

    
    # download image and optionally save it
    def getimage(self,img_url,dosave=True):
        imgraw = requests.get(img_url,headers=self.headers,timeout=5)
        # screen for mistaken url that is not an image
        if 'html' in str(imgraw.content):
            warnings.warn('getimage() content is html, skipping...')
            return None,None
        img_ext = os.path.splitext(img_url)[1]
        if not img_ext:
            # imghdr deprecated, and a bytes object is not a byte stream, still easiest
            # way to handle this
            with open(self.fname,'wb') as fp:
                fp.write(imgraw.content)
            img_ext = '.'+imghdr.what(self.fname)
            os.remove(self.fname)
        if dosave:
            with open(os.path.join(self.fname+img_ext),'wb') as fp:
                fp.write(imgraw.content)
        img = Image.open(io.BytesIO(imgraw.content))
        # rturns PIL Image
        return img,img_ext
    # save it later. PIL Image or raw
    def saveimage(self,img,img_ext):
        if 'PIL' in str(type(img)):
            img.save(os.path.join(self.fname+img_ext))
        else:
            with open(os.path.join(self.fname+img_ext),'wb') as fp:
                fp.write(img.content)


    # handle variations and tidy up the url syntax
    def process_url(self,img_url,b):
        img_url = img_url.lstrip()
        img_url = re.sub(r'%3A',':',img_url)
        img_url = re.sub(r'%2F','/',img_url)
        if 'http' in img_url:
            img_url = re.sub(r'^.*http','http',img_url)
        elif img_url.startswith('//'):
            img_url = 'https:'+img_url
        elif img_url.startswith('/'):
            img_url = 'https://' + b['link'].split('/')[2] + img_url
        else:
            warnings.warn('url prepending failed, skipping...')
            return None
        # additionally remove any trailing characters after these common exts
        if any(ext in img_url for ext in [r'.gif',r'.png',r'.jpg',r'.jpeg']):
            img_url = re.sub(r'(\.gif|\.png|\.jpg|\.jpeg).*$',r'\1',img_url)
        else:
            # raise Exception('no image extension in image url')
            warnings.warn('No .extension in image url ')
            # precaliber. has jpg but not .jpg. if cutting at jpg, lose image size params present in the url 
            # after jpg, and get a small image. but image size params also include a space which breaks the request. 
            # cut at that space, keep most of the image size params and it works
            if ' ' in img_url:
                img_url = img_url.split(' ')[0]
        print(img_url)
        return img_url


    # process srcset attribute for the largest image, or take 'src' as default
    def process_srcset(self,p,i,defkey='src'):
        llink = i.attrs.get(defkey,'') # default
        # starting this off as another hack for prevelo zulu
        for k in ['srcset','data-srcset','data-src']:
            if k in i.attrs.keys():
                if not re.search(p,i.attrs[k],flags=re.I):
                    continue
            else:
                continue

            # a lot of url's have the calendar year as well, which is similar to a
            # typical large image size so might lose a match this way, but exclude them as follows
            # also excluding more than 4 digit strings.
            size_pattern = '2020|2021|2022|2023|[0-9]{5,}|([0-9]{3,4})'
            if '\n' in i.attrs[k]:
                llist = i.attrs[k].split('\n')
            elif ',' in i.attrs[k]:
                llist = i.attrs[k].split(',')
            else:
                llist = i.attrs[k].split(' ')
            lmax = 0
            for l in llist:
                # in order to find the largest image in the srcset, want to 
                # compare only the image filename, as there could be 4 digit
                # numbers in the path that are not sizes
                # this could exclude some true sizes that are part of the path
                # and not the filename
                lsplit = l.split('/')
                # find the path element that is the filename. it would usually
                # be the last but not taking any chance
                ltest = [x for x in lsplit if re.search(p,x,flags=re.I)]
                if len(ltest):
                    ltest = ltest[0]
                else:
                    ltest = l
                # get all the 3,4 digit numbers
                lmatch = list(filter(None,re.findall(size_pattern,ltest)))
                if len(lmatch):
                    # update the largest value
                    lsize = max([int(l2) for l2 in lmatch])
                    if lsize > lmax:
                        lmax = lsize
                        llink = l
        # starting this off as a hack try to make it more general
        if r'{width}' in llink:
            warnings.warn(r'attempting substitution {width}')
            if 'widths' in i.attrs.keys():
                w = max(literal_eval(i.attrs['widths']))
                llink = re.sub(r'{width}',str(w),llink)
            elif 'data-widths' in i.attrs.keys():
                w = max(literal_eval(i.attrs['data-widths']))
                llink = re.sub(r'{width}',str(w),llink)
            else:
                warnings.warn(r'failed substitution {width}')

        return llink


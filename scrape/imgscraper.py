from urllib3.exceptions import SSLError
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import glob
import re
import warnings
import requests
import numpy as np
from ast import literal_eval
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

    # process a list of profile image urls and take largest image
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
    



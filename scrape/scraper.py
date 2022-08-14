import os
import io
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
from process.profile import Profile
from process.process import Process


class Scraper():
    def __init__(self,debug=True):
        UAS = ("Mozilla/5.0 (Windows NT 10.0; WOW64; rv:102.0) Gecko/20100101 Firefox/102.0", 
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5; rv:102.0) Gecko/20100101 Firefox/102.0",
            "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0"
            )
        self.ua = UAS[random.randrange(len(UAS))]
        self.headers = {'user-agent': self.ua}
        self.session = requests.Session()
        self.debug = debug
        self.year = date.today().year
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
        soup = BeautifulSoup(ptext,'html.parser')
        imgs = soup.find_all('img')
        meta = soup.find_all('meta')
        links = soup.find_all('a')
        urls = []

        # form a short list of candidate images
        if len(imgs) > 0:
            urls = self.get_imgurls(imgs)
        if len(meta) > 0:
            meta_urls = self.get_metaurls(meta)
            if meta_urls is not None:
                urls.append(meta_urls)
        if len(urls) == 0:
            warnings.warn('No img or meta urls found. continuing')
            return

            # some problem cases
            # norco images served by jscrip? but hi-res appear hidden in meta
            # commencal served by jscrip? but only a med-res available in meta
            # lone peak profile is in meta, but has other images in img tags
            # offair5 profile image is in a link. off5 pofile image is in meta
            # recon no label-pattern anywhere, meta, alt or img

        # go through the list and take the largest profile image
        profileimg_url,ext = self.search_urls(urls,b)

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
    def get_imgurls(self,imgs):
        urls = []
        # start with build pattern, fallback on model pattern
        for p in [self.build_pattern,self.model_pattern]:
            for i in imgs:
                # most generally look for name tag in the src attribute
                if re.search(p,i.attrs.get('src',''),flags=re.I): # khs alite
                    img_url = self.process_srcset(p,i)
                    urls.append(img_url)
                    continue
                
                # also try the data-src attribute

                # early seeker has src that is small, data-src with widths.
                # cdale trail has a src masking the larger image in data-src
                # Giant STP,XTC has no src, small data-src, large image jscrip served?? but in browser,
                # a larger image is served. another missing header??

                if re.search(p,i.attrs.get('data-src',''),flags=re.I): # khs alite
                    img_url = self.process_srcset(p,i,defkey='data-src')
                    urls.append(img_url)
                    continue

                # next try the alt attribute

                # tairn has label in alt tag, not image src name
                if re.search(p,i.attrs.get('alt',''),flags=re.I): # tairn
                    # dback pines doesn't have name in the 'src' attr, but it can be 
                    # inferred by a data-widths attr. meanwhile, lack of a lazyload
                    # method blocks the correct assignment. so this is just a kludge
                    # for now may or may not need to be permanent.
                    if 'data-widths' in i.attrs.keys():
                        img_url = self.process_srcset(p,i)
                        urls.append(img_url)
                        continue
                    # not sure whether to use lazyload or not
                    # for MTB 69, it is needed to get the largest image
                    if 'lazyload' in i.attrs.get('class',''):
                        # TODO method to process a lazyload class
                        continue
                    # TODO handle variations on 'src'
                    if 'src' in i.attrs.keys():
                        img_url = i.attrs['src']
                        urls.append(img_url)
                        continue

                # other problem cases
                # cleary scout may be jscrip served? but the easily findable image
                # in a tag appears to be mislabelled. scout 26 also broken
                # creig 26 appears to be jscrip served? even though creig 24 was not
                # max 26 same. appears to be jscrip served even though max 24 was not
                # tairn will need afurther algorithm to collect all the images and pick the right one
                # vpace 26 can make no sense of imgs, they don't match what's on the page
                # amulet 24,26 maybe another header tag needed? page response seems wrong-different than browser
                # roscoe self.build_pattern is in meta, but this picks up a wrong image with self.model_pattern first

            if len(urls):
                break
        return urls

    # search meta tags for list of possible images
    def get_metaurls(self,meta):
        urls = []
        for m in meta:
            if m.attrs.get('content','').startswith("http"): # for now assume we have this
                if re.search(self.model_pattern,m.attrs['content'],flags=re.I):
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
                    if re.search(self.build_pattern,m.attrs['content'],flags=re.I):
                        img_url = m.attrs['content'] # for special riprock, no ext but
                                                        # can at least check build
                        urls.append(img_url) 

        return urls

    
    # set re patterns and other data for current bike
    def setpattern(self,b):
        self.fname = os.path.join('/home/src/kids-24-bikes/png/'+str(self.year),b['type'],b['label']+'_'+b['build'])
        self.fname = re.sub(' ','',self.fname)
        # other attributes
        self.pfile = os.path.join('/home/src/kids-24-bikes/tmp/page',b['type'],b['label']+'_'+b['build']+'.pkl')
        self.referer = 'https://'+b['link'].split('/')[2]
        self.link = b['link']
        self.model_pattern = b['label'].replace(' ','.{0,3}') # single space hard-coded,by convention in spreadsheet
        if b['build']:
            self.build_pattern = b['build'].replace(' ','{0,3}') # same
            # self.build_pattern = self.model_pattern+'.*?'+b['build']
            # got one fail with .*?. try limiting to a small number of characters. should mostly be 1 or 0
            self.build_pattern = self.model_pattern+'.{0,3}'+self.build_pattern
        else:
            self.build_pattern = self.model_pattern

    
    # download image and optionally save it
    def getimage(self,img_url,dosave=True):
        imgraw = requests.get(img_url,headers=self.headers,timeout=5)
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
            # precaliber is this case, left all the trailing chars
            warnings.warn('No extension in image url ')
        print(img_url)
        return img_url


    # process srcset attribute for the largest image, or take 'src' as default
    def process_srcset(self,p,i,defkey='src'):
        llink = i.attrs.get(defkey,'') # default
        # starting this off as another hack for prevelo zulu
        for k in ['srcset','data-srcset']:
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

    # download the page from the main link given in spreadsheet
    # page is also stored for debugging purposes
    def getpage(self):
        if self.debug and os.path.exists(self.pfile):
            fp = open(self.pfile,'rb')
            ptext = pickle.load(fp)
            fp.close()
        else:
            self.session.get(self.referer,headers=self.headers)
            # gtbicycles. missing intermediate cert probably can't ssl it
            # is referer still a thing in 2022?? was needed for vitus
            # may solve some earlier problems too? eg vpace 26
            self.headers['referer'] = self.referer
            page = self.session.get(self.link,headers=self.headers,timeout=5)
            if 'META NAME=\"robots\"' in page.text: #scott has this
                warnings.warn('Robot blocked page serve, skipping: {}'.format(self.link))
                return
            ptext = page.text
            if self.debug:
                fp = open(self.pfile,'wb')
                pickle.dump(ptext,fp)
                fp.close()
        return ptext


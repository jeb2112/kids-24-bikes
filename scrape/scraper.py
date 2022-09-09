import os
import io
import requests
import random
import re
import pickle
import warnings
import requests
import imghdr
from PIL import Image
from datetime import date
import time
from ast import literal_eval

# base class for scraping
class Scraper():
    def __init__(self,debug=True,scrapedir=None):
        UAS = ("Mozilla/5.0 (Windows NT 10.0; WOW64; rv:102.0) Gecko/20100101 Firefox/102.0", 
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5; rv:102.0) Gecko/20100101 Firefox/102.0",
            "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0"
            )
        self.ua = UAS[random.randrange(len(UAS))]
        self.headers = {'user-agent': self.ua}
        self.session = requests.Session()
        self.year = date.today().year
        self.profiler = None
        self.processor = None
        self.debug = debug
        self.scrapedir = scrapedir
        return

    #############
    # main method
    #############

    def dosoup(self,b,debug=True):
        pass

    #############
    # aux methods
    #############

    # process a list of image urls and take largest image
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
            if w*h > maxsize:
                maxsize,maximgurl = w*h,imgurl
            if maxsize > 1e6: # big enough, avoid hammering website
                break
        return maximgurl,ext

    # get a list of candidate image urls
    # p. target string to search for
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

        return urls
    
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


    # download an image and optionally save it
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


    # search meta tags for list of possible targets
    def get_metaurls(self,meta,p):
        pass

    # set re patterns and other data for current bike
    def setpattern(self,b):
        self.fname = os.path.join('/home/src/kids-24-bikes',self.scrapedir,str(self.year),b['type'],b['label'])
        if not os.path.exists(os.path.dirname(self.fname)):
            os.makedirs(os.path.dirname(self.fname),exist_ok=True)
        if len(b['build']):
            self.fname += '_'+b['build']
        self.fname = re.sub(' ','',self.fname)
        # other attributes
        # temp store files for debugging and dev only
        self.pfile = os.path.join('/home/src/kids-24-bikes/tmp/page',b['type'],b['label'])
        if len(b['build']):
            self.pfile += '_'+b['build']
        self.pfile += '.pkl'
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

    # download the page from the main link given in spreadsheet
    # page is also stored for debugging purposes
    def getpage(self):
        if self.debug and os.path.exists(self.pfile):
            fp = open(self.pfile,'rb')
            ptext = pickle.load(fp)
            fp.close()
        else:
            try:
                self.session.get(self.referer,headers=self.headers)
            except requests.exceptions.SSLError as e:
                # gtbicycles. missing intermediate cert probably can't ssl it
                print(e)
                return None
            time.sleep(0.2)
            self.headers['referer'] = self.referer
            page = self.session.get(self.link,headers=self.headers,timeout=5)
            if 'META NAME=\"robots\"' in page.text: #scott has this
                warnings.warn('Robot blocked page serve: {}'.format(self.link))
                # this attempt copying header from firefox browser didn't work
                if 'scott' in self.referer and False:
                    self.headers['origin'] = self.referer
                    self.headers['host'] = 'I.clarity.ms'
                    self.headers['Sec-Fetch-Site'] = 'cross-site'
                    self.headers['Sec-Fetch-Mode'] = 'cors'
                    self.headers['Sec-Fetch-Dest'] = 'empty'
                    page = self.session.get(self.link,headers=self.headers,timeout=5)
                return None
            ptext = page.text
            if self.debug:
                fp = open(self.pfile,'wb')
                pickle.dump(ptext,fp)
                fp.close()
        return ptext

    # get a list of candidate href links
    # p. target string to search for
    def get_hrefurls(self,links,p=None):
        urls = []
        if p == None:
            p = self.build_pattern

        for l in links:
            # most generally look for name tag in the src attribute
            lhref = l.attrs.get('href','')
            if re.search(p,lhref,flags=re.I): # khs alite
                urls.append(lhref)
                continue
        return urls
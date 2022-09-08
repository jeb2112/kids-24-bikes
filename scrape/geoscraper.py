from inspect import stack
from multiprocessing.sharedctypes import Value
import os
import io
import copy
from ssl import SSL_ERROR_EOF
from urllib3.exceptions import SSLError
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
import random
import glob
import re
import pickle
import json
import warnings
import requests
import imghdr
import numpy as np
from PIL import Image
from ast import literal_eval
from datetime import date
import difflib
import time
from process.profile import Profile
from process.process import Process
from scrape.scraper import Scraper

# class for getting geometry data
class GeoScraper(Scraper):
    def __init__(self,**kwargs):
        super(GeoScraper,self).__init__(self,**kwargs)
        # self.profiler = Profile(kfold=4,modelname='profile_model_adam',domodel=True)
        # self.processor = Process(pad=False)
        # dict of internal keys for possible keywords found in different websites
        # due to difficulty of handling both longform and shortform, shortforms are only keys
        # and longforms are only values. note therefore have duplicates on the shortforms as duplicate keys
        # reach and stack won't work if abbreviated to r and s.
        self.geolbls = {'rh':['reach'],'sk':['stack'],\
                        'st':['seattube','seattubelength'],\
                        'tt':['toptube','toptubeeffective','toptubelength'],\
                        'ht':['headtube','headtubelength'],\
                        'sa':['stangle','seattubeangle'],\
                        'sta':['stangle','seattubeangle'],\
                        'cs':['chainstay'],\
                        'rc':['rearcentre','rearcenter'],\
                        'bb':['bottombracketdrop','drop','bbdrop'],\
                        'ha':['htangle','headtubeangle'],\
                        'hta':['htangle','headtubeangle'],\
                        'fl':['forklength'],'fo':['forkoffset'],\
                        'wb':['wheelbase'],\
                        'soh':['soheight','standoverheight','standover'],\
                        'so':['soheight','standoverheight','standover'],'stm':['stem']}
        # dict of internal keys with default values
        self.geodflts = {'rh':350,'sk':500,\
                        'st':370,'tt':450,'ht':100,\
                        'hta':73,'ha':73,\
                        'sta':73,'sa':73,\
                        'cs':410,\
                        'rc':380,\
                        'bb':40,\
                        'fl':220,'fo':30,\
                        'wb':950,'soh':600,'so':600,'stm':70}
        # dict of internal keys for possible keywords in different websites
        self.complbls = {'lb':['weight','lb'],'frame':['frame'],'fork':['fork'],'crankarm':['cranklength','crankarm'],\
                        'rd':['rearderailleur','rearderailer','rd'],'spring':['spring'],\
                    'drive':['drive'],'brake':['brake'],'shifter':['shifter']}
        self.geodata = {}
        self.compdata = {}
        return

    #############
    # main method
    #############

    def dosoup(self,b,debug=True):

        self.setpattern(b)
        # for dev: if already have the photo don't hammer the server
        # TODO: check for fork with a different trail
        if glob.glob(self.fname+'.*'):
            warnings.warn('Photo already exists, returning...',stacklevel=2)
            return

        ptext = self.getpage()
        if ptext is None:
            warnings.warn('No page text retrieved, skipping...',stacklevel=2)
            return
        soup = BeautifulSoup(ptext,'html.parser')
        imgs = soup.find_all('img')
        meta = soup.find_all('meta')
        links = soup.find_all('a')
        scrip = soup.find_all('script')
        tbls = []
        # check html table tag
        try:
            tbls = pd.read_html(ptext)
        except ValueError as e:
            warnings.warn('no table tags in html',stacklevel=2)
        # check for a json dict
        jtbls = self.get_scriptables(scrip)
        if len(jtbls) == 0:
            warnings.warn('no json tables in scrip tag',stacklevel=2)
            # output for debugging
            if len(tbls) == 0:
                with open(self.fname+'_.html','w') as fp:
                    fp.write(soup.prettify())
                return
        urls = []

        # start with html tables if any
        # tables are identified by the table values, 
        # but may also need model or build pattern??
        # problem cases
        # fuji page broken shows 12" column on a 24 bike page, no way to select 24 geom
        if len(tbls) > 0:
            # this function can probably be reduced
            geotbl = self.get_tbl(tbls)
            if geotbl is not None:
                geotbl = self.process_tbl(geotbl)
                res = self.read_tbl(geotbl)
            else:
                warnings.warn('no geom table in html tables',stacklevel=2)
                res = None
            if res is not None:
                self.save_tbl()
                return
            else:
                warnings.warn('html table not parsed.',stacklevel=2)
            # self.comps = self.get_comptbl(tbls)

        # then try json table. need get_jtbl variant if more than 1
        if len(jtbls) == 1:
            jtbls = pd.DataFrame.from_dict(jtbls)
            jtbls = self.process_tbl(jtbls)
            res = self.read_tbl(jtbls)
            if res is None:
                warnings.warn('json table not parsed.',stacklevel=2)
                return
        else:
            warnings.warn('No geom tables found. continuing',stacklevel=2)
            return

        # TODO: if no html or json tables, search for imgs of tables   
        # start with full build pattern, then fall back to model pattern
        # or may not need this loop for geo tables?
        # form a short list of candidate images
        tblimg_url = None
        for p in [self.build_pattern,self.model_pattern]:
            if len(imgs) > 0:
                urls = self.get_imgurls(imgs,p)
            if len(meta) > 0:
                meta_urls = self.get_metaurls(meta,p)
                if meta_urls is not None:
                    urls = urls + meta_urls
            if len(urls) == 0:
                warnings.warn('No img or meta urls found. continuing',stacklevel=2)
                return

            # go through the list and take the geometry table image
            tblimg_url,ext = self.search_urls(urls,b)
            if tblimg_url:
                break

            if tblimg_url is None:
                warnings.warn('No geometry table image found, skipping...',stacklevel=2)
                return
            self.saveimage(tblimg_url,ext)



    ##############
    # json methods
    ##############

    # process scrips for table data
    def get_scriptables(self,scrips):
        # this pattern for norc fluid, may need adjustment
        patrn = re.compile(r'\s+const\s+[a-zA-Z]+\s+=\s+(\{.*?\});\n')
        # a shorter reliable list of keywords. or use self.geolbls values > len(3)
        ktest = ['Reach','Stack'] # case not handled yet
        for s in scrips:
            if s.string is None:
                continue
            if any(k in s.string for k in ktest):
                jstr = patrn.findall(s.string)
                for j in jstr:
                    if any(k in j for k in ktest):
                        jdata = json.loads(j)
                        for kt in ktest:
                            jdict = self.recurse_dict(jdata,kt,val=kt)
                            if jdict is not None:
                                return jdict
        return []

    # iteratively walk a json dict for one of a list of keys or value
    def recurse_dict(self,obj,key,val=None):
        if key in obj: 
            return obj[key]
        if len(val) and val in list(obj.values()): # not worked
            return obj
        for k, v in obj.items():
            if isinstance(v,dict):
                item = self.recurse_dict(v, key, val=val)
                if item is not None:
                    return item
            elif isinstance(v,list):
                for litem in v:
                    if isinstance(litem,dict):
                        if key in litem:
                            return litem[key]
                        # ie norco fluid has geom data in a list of dicts, one dict per geom measurement
                        # this will likely need generalizing
                        if len(val) and val in list(litem.values()):
                            return v
                        item1 = self.recurse_dict(litem,key,val=val)
                        if item1 is not None:
                            return item1



    ##############
    # table methods
    ##############

    # detection of labels has been separateed into long forms and abbreviations
    # to resolve some matching confusions
    # but still problems with reach,stack long versus short forms. this code may be needed
    # # sort to do stack reach first
    # dfrs = tbl.apply(lambda x:x.str.contains('reach|stack',regex=True)).any(axis=1)
    # tbl = pd.concat([tbl.loc[dfrs],tbl.loc[dfrs.apply(lambda x:not(x))]])
    # # make local copy list to work with
    # geolblskeys = list(self.geolbls.keys())
 
    def read_tbl(self,tbl):

        # start with labels col
        # dict to keep track of matches
        geolblcount = {}
        for col in tbl:
            # if a column with numbers detected, skip
            if any(type(t)==int or type(t)==float for t in tbl[col]):
                continue
            geocount = 0
            geolblcount[col] = geocount
            for k1 in self.geolbls.keys():
                for item in tbl[col]:
                    if item=='' or item==None:
                        continue
                    if type(item) == list:
                        item = item[0] # might need better
                    elif type(item) == dict or type(item) == bool:
                        break # skip this column
                    # long form, match key values
                    if len(item) > 3:
                        # exact match preferably
                        if any(x == item for x in self.geolbls[k1]):
                            geocount += 1
                        # partial match longform
                        elif any(x in item for x in self.geolbls[k1]):
                            geocount += 1
                    # short form, match keys
                    else:
                        # match shortform. only doing exact matches for now
                        if item == k1:
                            geocount += 1
            geolblcount[col] = geocount

        lblcol = max(geolblcount, key=lambda k:geolblcount[k])

        # map lbls in identified lblcolumn to standard labels
        locallbl = {}
        for lbl in tbl[lblcol]:
            for k1 in self.geolbls.keys():
                for glbl in self.geolbls[k1]:
                    # long form first
                    if lbl in glbl or glbl in lbl:
                        locallbl[lbl] = k1
                        break
                # short form. awkward kludge for reach stack
                if k1 in lbl and not any(x in lbl for x in ['reach','stack']):
                    locallbl[lbl] = k1
                

        # now assess each additional column as a possible data column
        # TODO: add some logic for the column names

        # make new cropped df to hold just fields and values
        # incredibly, this instantiation syntax from data arg fails empty in some cases
        try:
            croptbl = pd.DataFrame(columns=['field'],data=tbl[lblcol])
            if croptbl.empty:
                raise ValueError
        except ValueError:
            croptbl = tbl[[lblcol]] # note syntax. requires explicit list to make df, otherwise its series
            croptbl.columns = ['field']
        croptbl['values'] = 0
        max_valerror = 1e6
        for col in tbl.loc[:,lblcol:]:
            geovals = {}
            val_error = 0
            if col == lblcol:
                continue
            for l in locallbl.keys():
                # get item from df according to the label2key mapping
                # according to doc these are equivalent syntax, but they are not
                try:
                    item = tbl.query('@lblcol ==  @l')[col].iloc[0]
                except KeyError:
                    item = tbl[tbl[lblcol] == l][col].iloc[0]
                if type(item) == list:
                    item = item[0] # might need better

                if type(item) == str:
                    # CubScout. check for multiple items in field. hardcoded take the first one only
                    item = item.split(' ')[0]
                    if re.match('[0-9.-]+',item):
                        # re syntax. set negation ^ is inside set brackets. and speicals aren't escaped inside sets
                        item = re.sub('[^\d.-]','',item)
                        try:
                            val = float(item)
                        # failure to convert could mean an improper match to a table item that isn't a geom quantity
                        # otherwise don't have anything to catch a non-geom item in a geom table
                        except ValueError:
                            continue
                    else:
                        val = 0
                elif type(item) == np.int64:
                    val = int(item)
                elif type(item) == float or type(item) == int or type(item) == np.float64:
                    val = item
                if pd.isna(val):
                    val = 0
                val_error += abs((val - self.geodflts[locallbl[l]]))
                geovals[locallbl[l]] = val
            if val_error < max_valerror:
                datacol = col
                self.geodata = geovals
                max_valerror = val_error

        croptbl['values'] = tbl[datacol]

        return croptbl

        
    # process string values in table for consistency. may need more processing...
    def process_tbl(self,tbl):
        for col in tbl:
            if all(type(t) == str for t in tbl[col]):
                tbl[col] = tbl[col].apply(lambda x:x.lower().replace(' ',''))
        return tbl

    # get geo table from list of candidate tables
    def get_tbl(self,tbls):
        for i,t in enumerate(tbls):
            for col in t:
                # skip any columns with numbers
                if any(type(t)==int for t in t[col]):
                    continue
                if any(type(t)==float for t in t[col]):
                    continue
                lcol = []
                for i in range(len(t[col])):
                    lcol.append(t[col][i].lower().replace(' ',''))
                # detect type of table by number of matches to the expected keys
                # in any given column. 
                # first try, simple substring in
                compcount = 0
                for k1 in self.complbls.keys():
                    for k2 in lcol:
                        if any(k2 in x or x in k2 for x in self.complbls[k1]):
                            compcount += 1
                geocount = 0
                for k1 in self.geolbls.keys():
                    for k2 in lcol:
                        if any(k2 in x or x in k2 for x in self.geolbls[k1]):
                            geocount += 1
                # alternate try, difflib. didn't work usefully
                if False:
                    compdiff = 0
                    geodiff = 0
                    for l2 in lcol:
                        compdiff += len(difflib.get_close_matches(l2,self.complbls.keys(),n=1))
                        geodiff += len(difflib.get_close_matches(l2,self.geolbls.keys(),n=1))
                    compdiff /= len(lcol)
                    geodiff /= len(lcol)

                if compcount > geocount:
                    compcol = col
                    comptbl = t
                elif geocount > compcount:
                    geocol = col
                    geotbl = t
                    return t.loc[:,col:]
                
                a=1
                
        return None

    # save geomtables
    def save_tbl(self):
        if not os.path.exists(os.path.dirname(self.fname)):
            os.makedirs(os.path.dirname(self.fname),exist_ok=True)
        if True:
            fname = self.fname + '_geom.json'
            with open(fname,'w') as fp:
                json.dump(self.geodata,fp,indent=4)
        else:
            fname = self.fname + '_geom.pkl'
            with open(fname,'wb') as fp:
                pickle.dump(self.geodata,fp)

    #############
    # aux methods
    #############


    # methods from image scraping class.
    # to delete most of them
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

    def clear(self):
        self.geodata = {}
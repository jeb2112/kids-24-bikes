from inspect import stack
from multiprocessing.sharedctypes import Value
import os
import io
import requests
from bs4 import BeautifulSoup
import pandas as pd
import glob
import re
import json
import warnings
import requests
import imghdr
import numpy as np
from PIL import Image
from ast import literal_eval
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
        # head seat angles are listed first, so there will never be a mistaken assignment to heat seat tube
        self.geolbls = {'rh':['reach'],'sk':['stack'],\
                        'ha':['htangle','headtubeangle'],\
                        'hta':['htangle','headtubeangle'],\
                        'sa':['stangle','seattubeangle'],\
                        'sta':['stangle','seattubeangle'],\
                        'st':['seattube','seattubelength'],\
                        'tt':['toptube','toptubeeffective','toptubelength'],\
                        'ht':['headtube','headtubelength'],\
                        'cs':['chainstay'],\
                        'rc':['rearcentre','rearcenter'],\
                        'bb':['bottombracketdrop','drop','bbdrop'],\
                        'bbh':['bottombracketheight','bbheight'],\
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
                        'bbh':260,\
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
            warnings.warn('Geom already exists, returning...',stacklevel=2)
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
        # collect all available tables
        # check html table tag
        try:
            # pandas better than soup for tables
            tbls = pd.read_html(ptext)
        except ValueError as e:
            warnings.warn('no table tags in html',stacklevel=2)
        # check for any json table dicts in a script
        jtbls = self.get_scriptables(scrip)
        if len(jtbls) == 0:
            warnings.warn('no json tables in scrip tag',stacklevel=2)
            # output for debugging
            if len(tbls) == 0:
                with open(self.fname+'_.html','w') as fp:
                    fp.write(soup.prettify())
                return

        # parse the tables
        # start with html tables if any
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
            else:
                return
        else:
            warnings.warn('No geom tables found.',stacklevel=2)

        # last, try for an image of geom table in an image or link
        urls = []
        tblimg_url = None
        urlimgset = {}
        urllinkset = {}
        for p in ['geo',self.model_pattern,'geometry']:
            if len(imgs) > 0:
                urlimgset[p] = set(self.get_imgurls(imgs,p))
            if len(links) > 0:
                urllinkset[p] = set(self.get_hrefurls(links,p))
        # search first for any match in both geo, model
        urls += urlimgset['geo'] & urlimgset[self.model_pattern]
        urls += urllinkset['geo'] & urllinkset[self.model_pattern]
        # then any single match in geometry
        urls += list(urlimgset['geometry']) + list(urllinkset['geometry'])
        # then the set of geo
        urls += list(urlimgset['geo']) + list(urllinkset['geo'])
        urls = list(set(urls))

        if len(urls) == 0:
            warnings.warn('No img or link urls found for geom. continuing',stacklevel=2)
            return
        else:
        # go through the list and take the largest possible geometry table image
        # search_urls() probably to base class.
            tblimg_url,ext = self.search_urls(urls,b)
            if tblimg_url is None:
                warnings.warn('No geometry table image found, skipping...',stacklevel=2)
                return
        self.saveimage(tblimg_url,ext)
        return


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

    # parse table into dataframe. also populates a dict
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
        kflag = False
        lflag = False
        for lbl in tbl[lblcol]:
            # long form first
            for k1 in self.geolbls.keys():
                for glbl in self.geolbls[k1]:
                    if lbl in glbl or glbl in lbl:
                        locallbl[lbl] = k1
                        kflag = True
                        if not any(x in lbl for x in ['reach','stack']):
                            lflag = True # once a long form is matched, assume there are no short forms in table.
                        break
                if kflag:
                    kflag = False
                    break
            # short form. only check if no longform matches. this is another ad hoc check on non
            # geom items in a geom table. kludge for reach stack should be unnecessary
            if not lflag:
                if lbl not in locallbl.keys():
                    for k1 in self.geolbls.keys():
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
                    # pineridge. split on left bracket for in/mm, but need proper in/mm detection. eg zulu
                    item = item.split('(')[0]
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
    # counts matches on the various geom fieldnames to determine if the
    # table is a geom table
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
        # if not os.path.exists(os.path.dirname(self.fname)):
        #     os.makedirs(os.path.dirname(self.fname),exist_ok=True)
        fname = self.fname + '_geom.json'
        with open(fname,'w') as fp:
            json.dump(self.geodata,fp,indent=4)

    ###############
    # image methods
    ###############

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


    def clear(self):
        self.geodata = {}
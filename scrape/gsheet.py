from distutils.log import debug
import pygsheets
import pandas as pd
import re
import os
import pickle
import requests
import warnings
import random
import timeit
import glob
import imghdr
from ast import literal_eval
from bs4 import BeautifulSoup

#authorization

class Gsheet():
    def __init__(self,debug=True,online=True):

        self.gc = pygsheets.authorize(service_file='/home/src/kids-24-bikes/auth/kids-24-bikes-a9e2eea94e79.json')
        # ids = self.gc.spreadsheet_ids()
        self.bikes = []
        self.sh = None
        #open the google spreadsheet
        # somehow title broken, can't find
        # self.sh = self.gc.open("Kids 24\" bikes")
        # use hard-coded key from url instead
        if online:
            try:
                start = timeit.timeit()
                print('start')
                self.sh = self.gc.open_by_key('1FodMz3A9-ehyC2bdrjs72kN5OgBBWe_H9EtYCfVpBu0')
                print('end')
                end = timeit.timeit()
                print('time to open: {}'.format(end-start))
            except pygsheets.SpreadsheetNotFound:
                warnings.warn('spreadsheet not found')

            # update the link to the video for each sheet
            self.wkss = self.sh.worksheets()
            self.title = '2022'
            self.wks = self.sh.worksheet_by_title(self.title)

        if debug and os.path.exists('/home/src/kids-24-bikes/tmp/bikes.pkl'):
            self.loadbikes()
        elif online:
            celldata = self.wks.get_row(3,value_render='FORMULA') # hard-coded 3rd row
            celldata = celldata[3:celldata.index('mean')] # hard-coded start 3rd col
            builddata = self.wks.get_row(4) # build string
            builddata = builddata[3:len(celldata)]
            for c,b in zip(celldata,builddata):
                bike = dict.fromkeys(['link','label','build'],[])
                m =re.search(r'=HYPERLINK\("(.*?)","(.*?)"\)',c)
                bike['link'] = m.group(1)
                bike['label'] = m.group(2)
                bike['build'] = b
                self.bikes.append(bike)

            if debug:
                self.savebikes()
        else:
            warnings.warn('no spreadsheet access')
        return

    def dosoup(self,b,debug=True):
        # user agent is still a thing in 2022. find a better place for this
        UAS = ("Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1", 
            "Mozilla/5.0 (Windows NT 6.3; rv:36.0) Gecko/20100101 Firefox/36.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10; rv:33.0) Gecko/20100101 Firefox/33.0",
            "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36",
            )
        ua = UAS[random.randrange(len(UAS))]
        headers = {'user-agent': ua}

        img_fname = os.path.join('/home/src/kids-24-bikes/png/2022',b['label'])
        img_fname = re.sub(' ','',img_fname)
        # for multiple builds, if already have one photo don't repeat
        # TODO: check for fork with a different trail
        if glob.glob(img_fname+'.*'):
            return

        model_pattern = b['label'].replace(' ','.*?') # single space hard-coded
        if b['build']:
            build_pattern = model_pattern+'.*?'+b['build']
        else:
            build_pattern = model_pattern
        pfile = os.path.join('/home/src/kids-24-bikes/tmp/page',b['label']+'.pkl')
        if debug and os.path.exists(pfile):
            fp = open(pfile,'rb')
            ptext = pickle.load(fp)
            fp.close()
        else:

            http_session = requests.Session()
            # http_session.get('https://'+b['link'].split('/')[2],headers=headers)
            # gtbicycles. missing intermediate cert probably can't ssl it
            page = http_session.get(b['link'],headers=headers,timeout=5)
            if 'META NAME=\"robots\"' in page.text: #scott i think had this
                warnings.warn('Robot blocked page serve, skipping: {}'.format(b['link']))
                return
            ptext = page.text
            if debug:
                fp = open(pfile,'wb')
                pickle.dump(ptext,fp)
                fp.close()
        soup = BeautifulSoup(ptext,'html.parser')
        imgs = soup.find_all('img')
        meta = soup.find_all('meta')
        links = soup.find_all('a')
        img_url = None
        if len(imgs) == 0: # norco images served by jscrip? but hi-res appear hidden in meta
            # commencal served by jscrip? but only a med-res available in meta
            # lone peak profile is in meta, but has other images in img tags
            img_url = self.process_meta(meta,model_pattern,build_pattern)
            # offair5 profile image is in a link
            # recon no label-pattern anywhere, meta, alt or img

        else:
            # start with build pattern, fallback on model pattern
            for p in [build_pattern,model_pattern]:
                for i in imgs:
                    # most generally look for name tag in the src attribute
                    # start with build_pattern, fall back on model_pattern
                    if re.search(p,i.attrs.get('src',''),flags=re.I): # khs alite
                        img_url = self.process_srcset(p,i)
                        break # TODO. iterate all imgs and take the largest one
                    # pello reyes has data-src, no src
                    # cdale trail has a src masking the larger image in data-src
                    # Giant STP,XTC has no src, small data-src, large image served??
                    # next try the data-src attribute
                    elif re.search(p,i.attrs.get('data-src',''),flags=re.I): # khs alite
                        img_url = self.process_srcset(p,i,defkey='data-src')
                        break
                    # tairn has label in alt tag, not image src name
                    # tairn will need afurther algorithm to collect all the images and pick the right one
                    # next try the alt attribute
                    elif re.search(p,i.attrs.get('alt',''),flags=re.I): # tairn
                        if 'lazyload' in i.attrs.get('class',''): # quick hack for lazyload thing
                            continue
                        img_url = i.attrs['src']
                        break
                    # special cases
                    # flowdown doesn't appear in src, but does in a wrong image, but could
                    # use data-widths > 1 as a clue to the unlabelled images
                    # cleary scout may be jscrip served? but the easily findable image
                    # in a tag appears to be mislabelled.
                    # this case for mec ace, but try to generalize it
                    elif 'Ace' in b['label'] and 'px' in i.attrs.get('sizes',''):
                        img_url = self.process_srcset(i)
                        break
                if img_url:
                    break
            # if no matches, try the meta tags
            if img_url is None:
                img_url = self.process_meta(meta,model_pattern,build_pattern)
        img_url = img_url.lstrip()
        img_url = re.sub(r'%3A',':',img_url)
        img_url = re.sub(r'%2F','/',img_url)
        if img_url.startswith('//'):
            img_url = 'https:'+img_url
        elif img_url.startswith('/'):
            img_url = 'https://' + b['link'].split('/')[2] + img_url
        elif 'http' in img_url:
            img_url = re.sub(r'^.*http','http',img_url)
        else:
            raise Exception('url prepending failed.')
        # additionally remove any trailing characters after these common exts
        if any(ext in img_url for ext in [r'.gif',r'.png',r'.jpg']):
            img_url = re.sub(r'(\.gif|\.png|\.jpg).*$',r'\1',img_url)
        else:
            # raise Exception('no image extension in image url')
            # precaliber is this case, left all the trailing chars
            warnings.warn('No extension in image url ')
        print(img_url)
        img = requests.get(img_url,headers=headers,timeout=5)
        img_ext = os.path.splitext(img_url)[1]
        if not img_ext:
            # imghdr deprecated, and a bytes object is not a byte stream, still easiest
            # way to handle this
            with open(img_fname,'wb') as fp:
                fp.write(img.content)
            img_ext = '.'+imghdr.what(img_fname)
            os.remove(img_fname)
        with open(os.path.join(img_fname+img_ext),'wb') as fp:
            fp.write(img.content)
        a=1

    def process_meta(self,meta,model_pattern,build_pattern):
        for m in meta:
            if m.attrs.get('content','').startswith("http"): # for now assume we have this
                if re.search(model_pattern,m.attrs['content'],flags=re.I):
                    if any(ext in m.attrs['content'] for ext in ['gif','jpg','png']):
                        img_url = m.attrs['content']
                        return img_url
                    else:
                        if re.search(build_pattern,m.attrs['content'],flags=re.I):
                            img_url = m.attrs['content'] # for special riprock, no ext but
                                                         # can at least check build
                            return img_url
                else: # for opus recon. risky but take any .png in a meta content
                    if any(ext in m.attrs['content'] for ext in ['gif','jpg','png']):
                        img_url = m.attrs['content']
                        return img_url

        return None

    # process srcset attribute for the largest image, or take 'src' as default
    def process_srcset(self,p,i,defkey='src'):
        llink = i.attrs[defkey] # default
        # starting this off as another hack for prevelo zulu
        for k in ['srcset','data-srcset']:
            if k in i.attrs.keys():
                if not re.search(p,i.attrs[k],flags=re.I):
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


    def getimage(self):
        pass

    def loadbikes(self):
        with open('/home/src/kids-24-bikes/tmp/bikes.pkl','rb') as fp:
            self.bikes = pickle.load(fp)
        # wcell = pygsheets.Cell('G1',val="",worksheet=self.wks)
        # cdata = wcell.get_json()
        pass

    def savebikes(self):
        with open('/home/src/kids-24-bikes/tmp/bikes.pkl','wb') as fp:
            pickle.dump(self.bikes,fp)
from distutils.log import debug
import pygsheets
import pandas as pd
import re
import os
import pickle
import requests
import warnings
from bs4 import BeautifulSoup

#authorization

class Gsheet():
    def __init__(self,debug=True):

        self.gc = pygsheets.authorize(service_file='/home/src/kids-24-bikes/auth/kids-24-bikes-a9e2eea94e79.json')
        ids = self.gc.spreadsheet_ids()
        self.bikes = [] 
        #open the google spreadsheet
        # somehow title broken, can't find
        # self.sh = self.gc.open("Kids 24\" bikes")
        # use hard-coded key from url instead
        self.sh = self.gc.open_by_key('1FodMz3A9-ehyC2bdrjs72kN5OgBBWe_H9EtYCfVpBu0')

        # update the link to the video for each sheet

        self.wkss = self.sh.worksheets()
        self.title = '2022'

        self.wks = self.sh.worksheet_by_title(self.title)

        if debug and os.path.exists('/home/src/kids-24-bikes/tmp/bikes.pkl'):
            self.loadbikes()
        else:
            celldata = self.wks.get_row(3,value_render='FORMULA') # hard-coded 3rd row
            celldata = celldata[3:celldata.index('mean')] # hard-coded start 3rd col
            for c in celldata:
                bike = dict.fromkeys(['link','label'],[])
                m =re.search(r'=HYPERLINK\("(.*?)","(.*?)"\)',c)
                bike['link'] = m.group(1)
                bike['label'] = m.group(2)
                self.bikes.append(bike)

            if debug:
                self.savebikes()
        return

    def dosoup(self,b,debug=True):
        label_pattern = b['label'].replace(' ','.*?') # single space hard-coded
        pfile = os.path.join('/home/src/kids-24-bikes/tmp/page',b['label']+'.pkl')
        if debug and os.path.exists(pfile):
            fp = open(pfile,'rb')
            ptext = pickle.load(fp)
            fp.close()
        else:
            page = requests.get(b['link'])
            ptext = page.text
            if debug:
                fp = open(pfile,'wb')
                pickle.dump(ptext,fp)
                fp.close()
        soup = BeautifulSoup(ptext,'html.parser')
        imgs = soup.find_all('img')
        if len(imgs) == 0: # norco images served by jscrip? but appear hidden in meta
            meta = soup.find_all('meta')
            img_url = self.process_meta(meta,label_pattern)

        else:
            for i in imgs:

                # mec ace
                if 'px' in i.attrs.get('sizes',''): # mec ace
                    img_url = self.process_srcset(i)
                    break
                # more generally look for name tag in the src attribute
                elif re.search(label_pattern,i.attrs.get('src',''),flags=re.I): # khs alite
                    img_url = self.process_srcset(i)
                    break

        img_url = re.sub(r'%3A',':',img_url)
        img_url = re.sub(r'%2F','/',img_url)
        if img_url.startswith('//'):
            img_url = 'https:'+img_url
        elif 'http' in img_url:
            img_url = re.sub(r'^.*http','http',img_url)
        else:
            raise Exception('url prepending failed.')
        if any(ext in img_url for ext in ['gif','png','jpg']):
            img_url = re.sub(r'(gif|png|jpg).*$',r'\1',img_url)
        else:
            raise Exception('no image extension in image url')
        print(img_url)
        img_ext = os.path.splitext(img_url)[1]
        img = requests.get(img_url)
        with open(os.path.join('/home/src/kids-24-bikes/png/2022',b['label']+img_ext),'wb') as fp:
            fp.write(img.content)
        a=1

    def process_meta(self,meta,label_pattern):
        for m in meta:
            if m.attrs.get('content','').startswith("http"): # for now assume we have this
                if re.search(label_pattern,m.attrs['content'],flags=re.I):
                    if any(ext in m.attrs['content'] for ext in ['gif','jpg','png']):
                        img_url = m.attrs['content']
                        return img_url
        return None

    # process srcset attribute for the largest image, or take 'src' as default
    def process_srcset(self,i):
        llink = i.attrs['src'] # default
        if 'srcset' in i.attrs.keys():
            # a lot of url's have the calendar year as well, which is similar to a
            # typical large image size so might lose a match this way, but exclude them as follows
            size_pattern = '2020|2021|2022|2023|([0-9]{3,4})'
            llist = i.attrs['srcset'].split('\n') # hard-coded newlin split here
            lmax = 0
            for l in llist:
                lmatch = list(filter(None,re.findall(size_pattern,l)))
                if len(lmatch):
                    lsize = max([int(l) for l in lmatch])
                    if lsize > lmax:
                        lmax = lsize
                        llink = l
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
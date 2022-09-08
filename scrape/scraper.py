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

    # process a list of urls for a target of some sort
    def search_urls(self,urls,b):
        pass

    # get a short list of candidate image urls
    def get_imgurls(self,imgs,p=None):
        pass
    
    # search meta tags for list of possible targets
    def get_metaurls(self,meta,p):
        pass

    # handle variations and tidy up the url syntax
    def process_url(self,img_url,b):
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


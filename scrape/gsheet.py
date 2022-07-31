from distutils.log import debug
import pygsheets
import pandas as pd
import re
import os
import pickle
import warnings
import timeit

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
            typedata = self.wks.get_row(1)
            typedata = typedata[3:len(celldata)]
            for c,b,t in zip(celldata,builddata,typedata):
                bike = dict.fromkeys(['link','label','build','type'],[])
                m =re.search(r'=HYPERLINK\("(.*?)","(.*?)"\)',c)
                bike['link'] = m.group(1)
                bike['label'] = m.group(2)
                bike['build'] = b
                if 'front' in t:
                    ctype = '24fs'
                elif '26' in t:
                    ctype = '26'
                elif '24' in t:
                    ctype = '24r'
                elif 'XS' in t:
                    ctype = 'xsl'
                bike['type'] = ctype
                self.bikes.append(bike)

            if debug:
                self.savebikes()
        else:
            warnings.warn('no spreadsheet access')
        return

    def loadbikes(self):
        with open('/home/src/kids-24-bikes/tmp/bikes.pkl','rb') as fp:
            self.bikes = pickle.load(fp)
        # wcell = pygsheets.Cell('G1',val="",worksheet=self.wks)
        # cdata = wcell.get_json()
        pass

    # save data to file to avoid hammering gsheets during debugging
    def savebikes(self):
        with open('/home/src/kids-24-bikes/tmp/bikes.pkl','wb') as fp:
            pickle.dump(self.bikes,fp)
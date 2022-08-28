from re import T
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
import os
import re
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections.abc import Iterable
from process.detector import Detector
import nltk

import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45,allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# detects geometry diagrams and/or table images
class GTableOCR():
    def __init__(self,kfold=1,modelname='gtableocr_model',epochs=50,domodel=False):
        self.batch_size = 4
        self.seed = 42
        self.num_classes = 2
        self.epochs = epochs

        self.x = []
        self.xtrn = []
        self.xval = []
        self.xtest = []
        self.y = 0
        self.npseed = 4

        self.words = []
        self.numbers = []
        self.tags = []

        self.maindir = '/home/src/kids-24-bikes/png/traindata/ocr'
        self.mdir = '/media/jbishop/WD4/kids-24-bikes/models'

        self.class0dir = os.path.join(self.maindir,'nongtable_ocr')
        self.class1dir = os.path.join(self.maindir,'gtable_ocr')
        self.loaddata()
        self.kfold = kfold
        self.k = 0
        self.nfold = int(len(self.y)/self.kfold)
        if domodel:
            self.model()

        self.modelname = modelname
        self.mdata = os.path.join(self.mdir,self.modelname+'_kfold{}_res.pkl'.format(self.kfold))
        self.sdata = np.zeros((self.kfold,6))
        self.ldata = np.zeros((self.kfold,self.epochs,2))
        self.fpidx = []
        self.fnidx = []


    def dotrain(self):
        self.fpidx = []
        self.fnidx = []
        for k in range(self.kfold):
            self.k = k
            self.traintest()
            classifier = nltk.NaiveBayesClassifier.train(self.trn)
            tdata = self.tally_predictions(classifier)
            self.sdata[k,:] = np.asarray(tdata[0:6])
            self.fnidx += tdata[6]
            self.fpidx += tdata[7]
        self.plot()


    # plot the results
    def plot(self):
        print('kFold{}'.format(self.kfold)+' Truth table')
        print('Predicted\tClass0\t\tClass1')
        print('0\t\t{:d}\t\t{:d}'.format(int(np.sum(self.sdata[:,1])),int(np.sum(self.sdata[:,3]))))
        print('1\t\t{:d}\t\t{:d}'.format(int(np.sum(self.sdata[:,2])),int(np.sum(self.sdata[:,0]))))
        print('Accuracy: {:.1f}'.format(100*np.mean(self.sdata[:,5])))
        # FP,FN cases
        fpfnlist = self.fpidx + self.fnidx
        nfp = len(self.fpidx)
        nplot = len(fpfnlist)
        nfn = nplot-nfp
        print('fp: {}'.format(self.fpidx))
        print('fn: {}'.format(self.fnidx))
        a=1

    def plot_features(self):
        fkeys=[]
        dictkeys = list(self.featuresets[0][0].keys())
        for i,f in enumerate(dictkeys):
            fkeys.append(re.search('[0-9]+',f).group(0))
        fpos = np.zeros((len(fkeys),))
        fneg = np.zeros((len(fkeys),))
        for f in self.featuresets:
            if f[1] == 1:
                fpos += np.array(list(f[0].values()))
            elif f[1] == 0:
                fneg += np.array(list(f[0].values()))
        barx = np.arange(np.shape(fpos)[0])
        w = 0.2
        fig,ax = plt.subplots()
        plt.bar(barx-w/2,fpos,w,label='pos')
        plt.bar(barx+w/2,fneg,w,label='neg')
        ax.set_xticks(barx,fkeys)
        ax.legend()
        plt.show()
        return

    def loaddata(self):
        self.raw_ds = tf.keras.preprocessing.text_dataset_from_directory(
                self.maindir,
                labels='inferred',
                shuffle=False,
                batch_size=1        )
        # dataset from dir seems to assume there are pre-separated dirs for train/test
        # this is awkward when all data is in 1 dir, so converting it to list
        # which is awkward, because determining the size of the datset in order to allocate
        # an np array, kind of invalidates the use of the numpy iterator
        self.ndata = self.raw_ds.cardinality().numpy()
        self.y = np.zeros((self.ndata,))
        self.raw_ds = self.raw_ds.enumerate()
        for i,e in self.raw_ds.as_numpy_iterator():
            txt = e[0][0].decode('utf-8').lower()
            tkns = nltk.word_tokenize(txt)
            self.x.append(tkns)
            self.y[i] = 1 - e[1][0]
        if False: # don't need categorical for nltk?
            self.y = keras.utils.to_categorical(self.y,self.num_classes)
        self.preprocess()
        # self.plot_features()
        return

    def features(self):
        for d in self.x:
            self.words += [w for w in d if re.fullmatch('[a-z]{1,}',w,flags=re.IGNORECASE)]
            self.numbers += [n for n in d if re.fullmatch('[0-9]+',n)]
            # self.tags += [t for t in d if re.fullmatch('[dp][0-9]+',t,flags=re.IGNORECASE)]
        # round numbers to single significant digit
        # self.numbers = [str(np.around(int(x),decimals=-(len(x)-1))) for x in self.numbers]
        # 1,2 letter words restricted to this list
        # self.words = [x for x in self.words if len(x)>1 or x in ['m','l','s']]
        # self.words = [x for x in self.words if len(x)>2 or x in ['cm','mm','bb','xs','xl','wb','ht','so']]
        # freq dist for word features
        all_words = nltk.FreqDist(self.flatten(self.words))
        all_numbers = nltk.FreqDist(self.flatten(self.numbers))
        self.word_features = list(all_words)[:50]
        self.num_features = list(all_numbers)[:50]

    def document_features(self,d):
        features = {}
        # for t in self.tags:
        #     if t in d:
        #         features['tag'] = t
        for t in self.num_features:
            features['contains({})'.format(t)] = d.count(t)
        for t in self.word_features:
            features['contains({})'.format(t)] = d.count(t)
        return features

    def preprocess(self):
        # randomize the data
        np.random.seed(self.npseed)
        self.idx = np.argsort(np.random.random(len(self.y)))
        self.x = [x for x,_ in sorted(zip(self.x,self.idx),key=lambda v:v[1])]
        self.y = self.y[np.argsort(self.idx)]
        # self.y = self.y[self.idx]
        # extract the corpus features and create feature_sets
        self.features()
        self.featuresets = [(self.document_features(d),lbl) for (d,lbl) in zip(self.x,self.y)]

    def traintest(self,testfrac=0.2,valfrac=0.0):
        rollk = self.k*self.nfold
        xyroll = self.list_roll(self.featuresets,rollk)
        trainfrac = 1-testfrac-valfrac
        ntrn = int(trainfrac*len(self.y))
        ntest = int(testfrac*len(self.y))
        nval = len(self.y)-ntrn-ntest
        self.trn = xyroll[:ntrn]
        self.val = xyroll[ntrn:(ntrn+nval)]
        self.test = xyroll[(ntrn+nval):]
        # make a note of where the test data start
        self.testidx0 = np.remainder(len(self.y)+ntrn+nval-rollk,len(self.y))

    # create the model
    def model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(self.rows,self.cols,1)))
        self.model.add(Conv2D(32,(3,3),activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        tf.keras.regularizers.L2(l2=0.001)
        self.model.add(Dense(32,activation='relu',kernel_regularizer='l2'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes,activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                            # optimizer=keras.optimizers.Adadelta(),
                            optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                            metrics=['accuracy'])
        print('Model parameters = %d'.format(self.model.count_params()))
        print(self.model.summary())
        self.model.save_weights(os.path.join(self.mdir,self.modelname+'_init_weights.h5'))


    def flatten(self,xs):
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from self.flatten(x)
            else:
                yield x

    def list_roll(self,data,shift):
        tmp = data[shift:]
        tmp += data[:shift]
        return tmp

    # create some results data
    def tally_predictions(self,classifier,doplot=False):
        # need better place for this
        ytestlabels = [x[1] for x in self.test]
        xtestsets = [x[0] for x in self.test]
        # p = np.argmax(self.model.predict(self.xtest,verbose=0),axis=-1)
        p = classifier.classify_many(xtestsets)
        score = nltk.classify.accuracy(classifier,self.test)
        print('Test accuracy: {}'.format(score))
        tp = tn = fp = fn = 0
        fnidx = []
        fpidx = []
        for i in range(len(ytestlabels)):
            if (p[i] == 0) and (ytestlabels[i] == 0):
                tn += 1
            elif (p[i] == 0) and (ytestlabels[i] == 1):
                # if doplot:
                #     self.plotimtest(i,tag='FN')
                #     a=1
                fnidx.append(i+self.testidx0)
                fn += 1
            elif (p[i] == 1) and (ytestlabels[i] == 0):
                # if doplot:
                #     self.plotimtest(i,tag='FP')
                #     a=1
                fpidx.append(i+self.testidx0)
                fp += 1
            elif (p[i] == 1) and (ytestlabels[i] == 1):
                tp += 1
        return tp,tn,fp,fn,0,score,fnidx,fpidx
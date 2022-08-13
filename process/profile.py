from re import T
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
import os
from PIL import Image
import imghdr
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections.abc import Iterable

import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45,allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# detects profile photos
class Profile():
    def __init__(self,kfold=1,modelname='profile_model',epochs=50,domodel=True):

        self.batch_size = 16
        self.num_classes = 2
        self.epochs = epochs
        self.nlist = []
        self.plist = []
        self.x = []
        self.xtrn = []
        self.xval = []
        self.xtest = []
        self.y = self.ytrn = self.yval = self.ytest = 0
        self.xtrn

        self.class0dir = '/home/src/kids-24-bikes/png/nonprofile_processed'
        self.class1dir = '/home/src/kids-24-bikes/png/profile_processed'
        self.mdir = '/media/jbishop/WD4/kids-24-bikes/models'

        self.kfold = kfold
        self.k = 0
        self.modelname = modelname
        self.mdata = os.path.join(self.mdir,self.modelname+'_kfold{}_res.pkl'.format(self.kfold))
        self.sdata = np.zeros((self.kfold,6))
        self.ldata = np.zeros((self.kfold,self.epochs,2))
        self.fpidx = []
        self.fnidx = []

        self.npseed = 4
        self.loaddata()
        self.nfold = int(len(self.y)/self.kfold)
        if domodel:
            self.model()

        if os.path.exists(self.mdata):
            with open(self.mdata,'rb') as fp:
                (self.sdata,self.ldata,self.fpidx,self.fnidx) = pickle.load(fp)

    # plot the results
    def plot(self):
        print('kFold{}'.format(self.kfold)+' Truth table')
        print('Predicted\tNonprofile\tProfile')
        print('N\t\t{:d}\t\t{:d}'.format(int(np.sum(self.sdata[:,1])),int(np.sum(self.sdata[:,3]))))
        print('P\t\t{:d}\t\t{:d}'.format(int(np.sum(self.sdata[:,2])),int(np.sum(self.sdata[:,0]))))
        print('Accuracy: {:.1f}'.format(100*np.mean(self.sdata[:,5])))
        # lossplots
        plt.figure(1)
        plt.plot(np.mean(self.ldata,axis=0))
        plt.legend(['train','val'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.ylim((0,1))
        # plt.show()
        # FP,FN images
        plt.figure(2,figsize=(10,2))
        fpfnlist = list(self.flatten(self.fpidx))
        nfp = len(fpfnlist)
        fpfnlist = fpfnlist + list(self.flatten(self.fnidx))
        nplot = len(fpfnlist)
        nfn = nplot-nfp
        for i in range(nplot):
            plt.subplot(1,nplot,i+1)
            plt.imshow(self.x[fpfnlist[i]],cmap='gray')
            plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
                            labelbottom=False,labelleft=False)
            if fpfnlist[i] in self.fpidx:
                plt.title('FP, {}'.format(fpfnlist[i]))
            else:
                plt.title('FN, {}'.format(fpfnlist[i]))
        plt.show()
        a=1

    # create the model
    def model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(self.rows,self.cols,1)))
        self.model.add(Conv2D(64,(3,3),activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        # extra conv didn't help much
        if False:
            self.model.add(Conv2D(64,(3,3),activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        if True:
            tf.keras.regularizers.L2(l2=0.001)
            self.model.add(Dense(64,activation='relu',kernel_regularizer='l2'))
        else:
            # fully convolutional is broken on tensor shape somehow
            self.model.add(Conv2D(128,(12,12),activation='relu'))
        self.model.add(Dropout(0.5))
        if True:
            self.model.add(Dense(self.num_classes,activation='softmax'))
        else:
            # fully convolutional
            self.model.add(Conv2D(2,(1,1),activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                            # optimizer=keras.optimizers.Adadelta(),
                            optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                            metrics=['accuracy'])
        print('Model parameters = %d'.format(self.model.count_params()))
        print(self.model.summary())
        self.model.save_weights(os.path.join(self.mdir,self.modelname+'_init_weights.h5'))

    # partition the data
    # k is the offset for the current fold
    def traintest(self,testfrac=0.1,valfrac=0.1):
        rollk = self.k*self.nfold
        xroll = np.roll(self.x,rollk,axis=0)
        yroll = np.roll(self.y,rollk,axis=0)
        trainfrac = 1-testfrac-valfrac
        ntrn = int(trainfrac*len(self.y))
        ntest = int(testfrac*len(self.y))
        nval = len(self.y)-ntrn-ntest
        self.xtrn = xroll[:ntrn]
        self.ytrn = yroll[:ntrn]
        self.xval = xroll[ntrn:(ntrn+nval)]
        self.yval = yroll[ntrn:(ntrn+nval)]
        self.xtest = xroll[(ntrn+nval):]
        self.ytest = yroll[(ntrn+nval):]
        # make a note of where the test data start
        self.testidx0 = np.remainder(len(self.y)+ntrn+nval-rollk,len(self.y))
        if False:
            self.plotimtest(3,tag='traintest')
            a=1
        
    # load image data. all in memory for now
    def loaddata(self):
        flist = os.listdir(self.class0dir)
        n0 = len(flist)
        im0 = Image.open(os.path.join(self.class0dir,flist[0]))
        self.cols,self.rows = im0.size
        self.ndata = np.zeros((n0,self.rows,self.cols,1))
        for i,f in enumerate(flist):
            fpath = os.path.join(self.class0dir,f)
            img_ext = '.'+imghdr.what(fpath)
            if img_ext != 'png':
                pass
            im = Image.open(fpath)
            self.ndata[i,:,:,0] = np.array(im)
        self.nlabels = np.zeros(n0,dtype=int)
        flist = os.listdir(self.class1dir)
        p0 = len(flist)
        self.pdata = np.zeros((p0,self.rows,self.cols,1))
        for i,f in enumerate(flist):
            fpath = os.path.join(self.class1dir,f)
            img_ext = '.'+imghdr.what(fpath)
            if img_ext != 'png':
                pass
            im = Image.open(fpath)
            self.pdata[i,:,:,0] = np.array(im)
        self.plabels = np.ones(p0,dtype=int)
        self.y = np.concatenate((self.nlabels, self.plabels),axis=0)
        self.y = keras.utils.to_categorical(self.y,self.num_classes)
        self.x = np.concatenate((self.ndata,self.pdata),axis=0)
        self.preprocess()

    # np array
    def preprocess(self):   
        # order
        if K.image_data_format() == 'channels_first':
            self.x = np.transpose(self.x,axes=(0,3,1,2))
        # scale the data
        self.x = self.x.astype('float32')
        self.x = self.x / 255
        # randomize the data
        np.random.seed(self.npseed)
        idx = np.argsort(np.random.random(len(self.y)))
        self.x = self.x[idx,:,:,:]
        self.y = self.y[idx]

    # reprediction
    def dopredict(self):
        self.fpidx = []
        self.fnidx = []
        for k in range(self.kfold):
            self.k = k
            self.traintest()
            filepath = os.path.join(self.mdir,self.modelname+'_kfold{:01d}'.format(self.k)+'_best.hdf5')
            self.model.load_weights(filepath)
            tdata = self.tally_predictions(doplot=False)
            self.sdata[k,:] = np.asarray(tdata[0:6])
            self.fnidx += tdata[6]
            self.fpidx += tdata[7]
        with open(self.mdata,'wb') as fp:
            pickle.dump((self.sdata,self.ldata,self.fpidx,self.fnidx),fp)
        self.plot()

    # test one image
    def test(self,img):
        p = np.zeros(self.kfold,dtype=int)
        for k in range(self.kfold):
            filepath = os.path.join(self.mdir,self.modelname+'_kfold{:01d}'.format(self.k)+'_best.hdf5')
            self.model.load_weights(filepath)
            p[k] = np.argmax(self.model.predict(img)) # or use the softmax value?
        p0 = int(np.round(np.mean(p)))
        return p0


    # main method for training
    def dotrain(self):
        self.fpidx = []
        self.fnidx = []
        for k in range(self.kfold):
            self.k = k
            self.traintest()
            self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath = os.path.join(self.mdir,self.modelname+'_kfold{:01d}'.format(self.k)+'_best.hdf5'),
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True)
            history = self.model.fit(self.xtrn,self.ytrn,
                                    batch_size = self.batch_size,
                                    epochs = self.epochs,
                                    verbose = 1,
                                    validation_data = (self.xval,self.yval),
                                    callbacks=[self.model_checkpoint])
            tdata = self.tally_predictions()
            self.sdata[k,:] = np.asarray(tdata[0:6])
            self.fnidx += tdata[6]
            self.fpidx += tdata[7]
            mpath = os.path.join(self.mdir,self.modelname+'_k{}_final.h5'.format(k))
            if False:
                self.model.save(mpath)
            self.model.load_weights(os.path.join(self.mdir,self.modelname+'_init_weights.h5'))
            self.ldata[k,:,0] = history.history['loss']
            self.ldata[k,:,1] = history.history['val_loss']
        with open(self.mdata,'wb') as fp:
            pickle.dump((self.sdata,self.ldata,self.fpidx,self.fnidx),fp)
        self.plot()

    # create some results data
    def tally_predictions(self,doplot=False):
        p = np.argmax(self.model.predict(self.xtest,verbose=0),axis=-1)
        score = self.model.evaluate(self.xtest,self.ytest,verbose=0)
        print('Test loss: {}'.format(score[0]))
        print('Test accuracy: {}'.format(score[1]))
        tp = tn = fp = fn = 0
        fnidx = []
        fpidx = []
        ytestlabels = np.argmax(self.ytest,axis=-1)
        for i in range(len(ytestlabels)):
            if (p[i] == 0) and (ytestlabels[i] == 0):
                tn += 1
            elif (p[i] == 0) and (ytestlabels[i] == 1):
                if doplot:
                    self.plotimtest(i,tag='FN')
                    a=1
                fnidx.append(i+self.testidx0)
                fn += 1
            elif (p[i] == 1) and (ytestlabels[i] == 0):
                if doplot:
                    self.plotimtest(i,tag='FP')
                    a=1
                fpidx.append(i+self.testidx0)
                fp += 1
            elif (p[i] == 1) and (ytestlabels[i] == 1):
                tp += 1
        return tp,tn,fp,fn,score[0],score[1],fnidx,fpidx

    # convenience/debugging methods
    def plotimtest(self,i,tag=None):
        plt.figure(10)
        plt.subplot(1,2,1)
        plt.imshow(self.xtest[i],cmap='gray',vmin=0,vmax=1)
        if tag is not None:
            plt.title(tag)
        plt.subplot(1,2,2)
        plt.imshow(self.x[i+self.testidx0],cmap='gray',vmin=0,vmax=1)
        plt.title('i={}, i+i0={}'.format(i,i+self.testidx0))
        plt.show()
        # plt.close()
        return

    def flatten(self,xs):
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from self.flatten(x)
            else:
                yield x
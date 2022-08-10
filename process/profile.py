import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
import os
from PIL import Image
import imghdr
import numpy as np

import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45,allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# detects profile photos
class Profile():
    def __init__(self):

        self.batch_size = 16
        self.num_classes = 2
        self.epochs = 100
        self.nlist = []
        self.plist = []
        self.x = self.xtrn = self.xval = self.xtest = []
        self.y = self.ytrn = self.yval = self.ytest = 0
        self.xtrn

        self.class0dir = '/home/src/kids-24-bikes/png/nonprofile_processed'
        self.class1dir = '/home/src/kids-24-bikes/png/profile_processed'

        self.loaddata()

        self.model = Sequential()
        self.model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(self.rows,self.cols,1)))
        self.model.add(Conv2D(128,(3,3),activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        if False:
            self.model.add(Conv2D(64,(3,3),activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        if True:
            self.model.add(Dense(64,activation='relu'))
        else:
            self.model.add(Conv2D(128,(12,12),activation='relu'))
        self.model.add(Dropout(0.5))
        if True:
            self.model.add(Dense(self.num_classes,activation='softmax'))
        else:
            self.model.add(Conv2D(2,(1,1),activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=['accuracy'])
        print('Model parameters = %d'.format(self.model.count_params()))
        print(self.model.summary())

    def traintest(self,testfrac=0.1,valfrac=0.1):
        trainfrac = 1-testfrac-valfrac
        ntrn = int(trainfrac*len(self.y))
        ntest = int(testfrac*len(self.y))
        nval = len(self.y)-ntrn-ntest
        self.xtrn = self.x[:ntrn]
        self.ytrn = self.y[:ntrn]
        self.xval = self.x[ntrn:(ntrn+nval)]
        self.yval = self.y[ntrn:(ntrn+nval)]
        self.xtest = self.x[(ntrn+nval):]
        self.ytest = self.y[(ntrn+nval):]
        
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
        # order
        if K.image_data_format() == 'channels_first':
            self.x = np.transpose(self.x,axes=(0,3,1,2))
        # scale the data
        self.x = self.x.astype('float32')
        self.x = self.x / 255
        # randomize the data
        idx = np.argsort(np.random.random(len(self.y)))
        self.x = self.x[idx,:,:,:]
        self.y = self.y[idx]

    def dotrain(self):
        self.traintest()
        history = self.model.fit(self.xtrn,self.ytrn,
                                batch_size = self.batch_size,
                                epochs = self.epochs,
                                verbose = 1,
                                validation_data = (self.xval,self.yval))
        score = self.model.evaluate(self.xtest,self.ytest,verbose=0)
        print('Test loss: {}'.format(score[0]))
        print('Test accuracy: {}'.format(score[1]))
        self.model.save('profile_model.h5')

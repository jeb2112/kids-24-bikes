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
from process.detector import Detector

import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.45,allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# detects profile photos
class Profile(Detector):
    def __init__(self,kfold=1,modelname='profile_model',epochs=50,domodel=True):
        super(Profile,self).__init__(kfold=kfold,modelname=modelname,epochs=epochs,domodel=domodel)

        self.class0dir = '/home/src/kids-24-bikes/png/trainingdata/processed/nonprofile_processed'
        self.class1dir = '/home/src/kids-24-bikes/png/trainingdata/processed/profile_processed'
        self.loaddata()
        self.nfold = int(len(self.y)/self.kfold)
        if domodel:
            self.model()


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
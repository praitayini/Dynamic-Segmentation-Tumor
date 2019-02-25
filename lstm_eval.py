#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:44:31 2018

@author: mira
"""

import os
import numpy as np
from definitions import *
from keras.backend import squeeze
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import ConvLSTM2D
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Concatenate as concatenate
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from scipy.io import savemat
import glob

NUM_CHANNELS=5
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
model = load_model('./logs/simple_lstm.h5')


launch=os.getcwd()



#validation=Dataset2D('/home/mira/karl_data/breast_tumor/in_data/core/Validation/',NUM_CHANNELS)
#NOTE THAT 'CHANNELS' IS SOMEWHAT MISLEADING IN THIS CODE SINCE WE ARE TALKING ABOUT TIME FRAMES NOT CHANNELS IN THE CONVENTIONAL SENSE OF THE WORD



testDir=launch + '/Testing/'

os.chdir(testDir)
testList  = glob.glob('*mat')

for i in testList:
    print(i)
    data=loadmat(i)
    mri=data['data']
    mri=mri[:,:,:,0:5]
    outt=np.zeros((mri.shape[0],mri.shape[1],mri.shape[2]))
    mask=mri[:,:,:,0]>15
    roi=data['roi_mat']
    roi=np.moveaxis(roi,2,0)
    mri=np.moveaxis(mri,3,0)
    mri=np.moveaxis(mri,3,0)
    mri=np.expand_dims(mri,2)
    a=model.predict_on_batch(mri[:,:,])
    out={}
    out['out']= a
    out['roi']= roi
    savemat('./matlab/'+i,out)

print('done :)')

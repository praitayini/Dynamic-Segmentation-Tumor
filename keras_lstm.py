import os
import numpy as np
from definitions import *
from keras import backend
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Lambda
from keras.layers import Input
from keras.layers import ConvLSTM2D
from keras.layers import Conv2D
from keras.backend import squeeze
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers import Permute
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
# fix random seed for reproducibility


launch=os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"]="1"

NUM_CHANNELS=5
BATCH_SIZE=16
IMG_HEIGHT=256
IMG_WIDTH=256
FILTERS=16
KERNEL_SIZE=3

training=Dataset2D('/home/petmri/training/',NUM_CHANNELS)
validation=Dataset2D('/home/petmri/Testing/',NUM_CHANNELS)
os.chdir(launch)
#NOTE THAT 'CHANNELS' IS SOMEWHAT MISLEADING IN THIS CODE SINCE WE ARE TALKING ABOUT TIME FRAMES NOT CHANNELS IN THE CONVENTIONAL SENSE OF THE WORD
#timeframe,row,column,channel

inputs = Input((NUM_CHANNELS,1,IMG_HEIGHT, IMG_WIDTH))
print(inputs.shape)
#print(inputs.__dict__)
squeezed = Lambda (lambda x: backend.squeeze(x,2))(inputs)
print(squeezed.shape)
l1 = ConvLSTM2D(FILTERS,KERNEL_SIZE,data_format='channels_first',padding='same',return_sequences=False) (inputs)
print(l1.shape)

merged = concatenate([squeezed,l1],axis=1)
#concat input with l1 output


#c1 = Conv2D(16, (1,1),data_format='channels_first',padding='same') (l1)
#c2 = Conv2D(16, (1,1),data_format='channels_first', padding='same') (c1)
#c3 = Conv2D(16, (1,1),data_format='channels_first', padding='same') (c2)
#c4 = Conv2D(16, (1,1),data_format='channels_first', padding='same') (c3)

c1 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first',  padding='same') (l1)
c1 = BatchNormalization(axis=1) (c1)
c1 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first', padding='same') (c1)
c1 = BatchNormalization(axis=1) (c1)
p1 = MaxPooling2D((2, 2),data_format='channels_first') (c1)
print(p1.shape)

c2 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first', padding='same') (p1)
c2 = BatchNormalization(axis=1) (c2)
c2 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first',  padding='same') (c2)
c2 = BatchNormalization(axis=1) (c2)
p2 = MaxPooling2D((2, 2),data_format='channels_first') (c2)
print(p2.shape)

c3 = Conv2D(256, (3, 3), activation='relu', data_format='channels_first', padding='same') (p2)
c3 = BatchNormalization(axis=1) (c3)
c3 = Conv2D(256, (3, 3), activation='relu', data_format='channels_first', padding='same') (c3)
c3 = BatchNormalization(axis=1) (c3)
p3 = MaxPooling2D((2, 2),data_format='channels_first') (c3)
print(p3.shape)

c4 = Conv2D(512, (3, 3), activation='relu', data_format='channels_first', padding='same') (p3)
c4 = BatchNormalization(axis=1) (c4)
c4 = Conv2D(512, (3, 3), activation='relu', data_format='channels_first', padding='same') (c4)
c4 = BatchNormalization(axis=1) (c4)
p4 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first') (c4)
print(p4.shape)

c5 = Conv2D(1024, (3, 3), activation='relu', data_format='channels_first', padding='same') (p4)
c5 = BatchNormalization(axis=1) (c5)
c5 = Conv2D(1024, (3, 3), activation='relu', data_format='channels_first', padding='same') (c5)
c5 = BatchNormalization(axis=1) (c5)
print(c5.shape)

u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), data_format='channels_first', padding='same') (c5)
u6 = concatenate([u6, c4], axis=1)
c6 = Conv2D(512, (3, 3), activation='relu', data_format='channels_first', padding='same') (u6)
c6 = BatchNormalization(axis=1) (c6)
c6 = Conv2D(512, (3, 3), activation='relu', data_format='channels_first', padding='same') (c6)
c6 = BatchNormalization(axis=1) (c6)
print(c6.shape)

u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), data_format='channels_first', padding='same') (c6)
u7 = concatenate([u7, c3],axis=1)
c7 = Conv2D(256, (3, 3), activation='relu', data_format='channels_first', padding='same') (u7)
c7 = BatchNormalization(axis=1) (c7)
c7 = Conv2D(256, (3, 3), activation='relu', data_format='channels_first', padding='same') (c7)
c7 = BatchNormalization(axis=1) (c7)
print(c7.shape)

u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), data_format='channels_first', padding='same') (c7)
u8 = concatenate([u8, c2],axis=1)
c8 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first', padding='same') (u8)
c8 = BatchNormalization(axis=1) (c8)
c8 = Conv2D(128, (3, 3), activation='relu', data_format='channels_first', padding='same') (c8)
c8 = BatchNormalization(axis=1) (c8)
print(c8.shape)

u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), data_format='channels_first', padding='same') (c8)
u9 = concatenate([u9, c1],axis=1)
c9 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first', padding='same') (u9)
c9 = BatchNormalization(axis=1) (c9)
c9 = Conv2D(64, (3, 3), activation='relu', data_format='channels_first', padding='same') (c9)
c9 = BatchNormalization(axis=1) (c9)
print(c9.shape)

outputs = Conv2D(1, (1, 1), data_format='channels_first', activation='sigmoid') (c9)


#merged = concatenate([inputs, outputs])

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
tensorboard.set_model(model)
checkpoint = ModelCheckpoint('logs/simple_lstm.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
checkpoint.set_model(model)

results = model.fit_generator(generator=training.lstm_generator_keras(BATCH_SIZE), epochs=1000, validation_data=validation.lstm_generator_keras(BATCH_SIZE), use_multiprocessing=True, steps_per_epoch=10, validation_steps=5,callbacks=[tensorboard, checkpoint])


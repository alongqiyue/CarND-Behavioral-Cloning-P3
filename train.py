# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:47:57 2017

@author: hds
"""

import csv
import cv2
import numpy as np
import utilties
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

epochs = 10
samplers_per_epoch = 400
validation_sampler = 80
DRIVING_FILE = 'data/driving_log.csv'

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout

model = Sequential()
model.add(Cropping2D(cropping=((70,20),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0)-0.5))
model.add(Convolution2D(24,5,5,activation="elu",subsample=(2,2)))
model.add(Convolution2D(36,5,5,activation="elu",subsample=(2,2)))
model.add(Convolution2D(48,5,5,activation="elu",subsample=(2,2)))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,activation="elu"))
model.add(Dense(50,activation="elu"))
model.add(Dense(10,activation="elu"))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

samples = utilties.get_csv_data(DRIVING_FILE)
train_samples, validation_samples = train_test_split(samples,test_size=0.2)

train_gen = utilties.generate_batch(train_samples)
valid_gen = utilties.generate_batch(validation_samples)

history_object = model.fit_generator(train_gen,epochs=10,steps_per_epoch = samplers_per_epoch,
                    validation_data=valid_gen,validation_steps=validation_sampler,
                    verbose = 1)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
#exit()
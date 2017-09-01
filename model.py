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

DRIVING_FILE = 'data/driving_log.csv'

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.layers import Dropout

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 127.5 - 1)) # image normalization function
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5)) # droupout with 50% keep prob added to reduce overfitting.
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()
model.compile(loss='mse',optimizer='adam')

samples = utilties.get_csv_data(DRIVING_FILE)
train_samples, validation_samples = train_test_split(samples,test_size=0.2)

train_gen = utilties.generate_batch(train_samples)
valid_gen = utilties.generate_batch(validation_samples)

history_object = model.fit_generator(train_gen, samples_per_epoch = len(train_samples), validation_data=valid_gen, nb_val_samples=len(validation_samples), nb_epoch=8, verbose=1)

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
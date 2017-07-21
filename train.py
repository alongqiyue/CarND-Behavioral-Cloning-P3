# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:47:57 2017

@author: hds
"""

import csv
import cv2
import numpy as np
cv2.calibrateCamera()
cv2.undistort()
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
del lines[0]
        
images = []
measurements = []

for line in lines:
    for i in range(3):    
        source_path =  line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)    
        if i==0:
            image_flip = np.fliplr(image)
            images.append(image_flip)
        
    correction = -0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(-measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)
x_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x/255.0)-0.5))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, validation_split=0.2,shuffle=True,epochs=10)

model.save('model.h5')
exit()
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:11:52 2017

@author: hds
"""
import csv
import numpy as np
import cv2
import sklearn.utils
from sklearn.model_selection import train_test_split
DRIVING_FILE = 'data/driving_log.csv'
CORRECTION = 0.23

def transform_csv_data(csv_file):
    lines = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            out_line = line
            for i in range(3):
                out_line[i] = 'IMG/' + out_line[i].split('\\')[-1]
            lines.append(out_line)        
    
    with open('hehe.csv') as csvfile:
        csv.writer()
    
    



def get_csv_data(csv_file):
    lines = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)        
    del lines[0]        
    return lines


def generate_batch(samples,batch_size = 64):
    num_samples = len(samples)
    while True:
        X_batch = []
        y_batch = []
        rnd_index = np.random.randint(0,num_samples,batch_size)
        for index in rnd_index:
            rnd_image = np.random.randint(0,3)
            image = cv2.imread('data/IMG/'+samples[index][rnd_image].split('/')[-1])        

            if rnd_image == 0:
                y_batch.append(float(samples[index][3]))
            elif rnd_image == 1:
                y_batch.append(float(samples[index][3])+CORRECTION)
            else:
                y_batch.append(float(samples[index][3])-CORRECTION)            
            X_batch.append(image)
        
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        return X_batch,y_batch
        
        
samples = get_csv_data(DRIVING_FILE)
train_samples, validation_samples = train_test_split(samples,test_size=0.2)

train_gen,y_gen = generate_batch(train_samples)
valid_gen = generate_batch(validation_samples)
        
        
        
        
        
        
        

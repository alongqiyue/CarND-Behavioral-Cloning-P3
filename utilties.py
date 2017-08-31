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
import random
DRIVING_FILE = 'data/driving_log.csv'
CORRECTION = 0.27

def transform_csv_data(csv_file):
    lines = []
    with open(csv_file) as file:
        reader = csv.reader(file)
        for line in reader:
            out_line = line
            for i in range(3):
                out_line[i] = 'IMG/' + out_line[i].split('\\')[-1]
            lines.append(out_line)        
    with open('hehe.csv','w',newline='') as outfile:
       writer = csv.writer(outfile)
       for line in lines:          
           writer.writerow(line)  
       outfile.close()

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
        random.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_batch = []
            y_batch = []
            for batch_sample in batch_samples:
                for i in range(3):
                    image = cv2.imread('data/IMG/'+batch_sample[i].split('/')[-1])   
                    angle = float(batch_sample[3])
                    if i == 1:
                        angle = angle + CORRECTION
                    elif i == 2:
                        angle = angle - CORRECTION

                    if angle < 0.001:
                        if np.random.random() < 0.5:
                            continue
                                            
                    X_batch.append(image)
                    y_batch.append(angle)
                    X_batch.append(np.fliplr(image))
                    y_batch.append(-angle)                    
                #rnd_image = np.random.randint(0,3)
                #image = cv2.imread('data/IMG/'+batch_sample[rnd_image].split('/')[-1])   
                #if rnd_image == 0:
                #    y_batch.append(float(batch_sample[3]))
                #elif rnd_image == 1:
                #    y_batch.append(float(batch_sample[3])+CORRECTION)
                #else:
                #    y_batch.append(float(batch_sample[3])-CORRECTION)    
                #X_batch.append(image)
                
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch,y_batch
            
#samples = get_csv_data(DRIVING_FILE)
#train_samples, validation_samples = train_test_split(samples,test_size=0.2)

#train_gen,y_gen = generate_batch(train_samples)
#valid_gen = generate_batch(validation_samples)
        
        
        
        
        
        
        

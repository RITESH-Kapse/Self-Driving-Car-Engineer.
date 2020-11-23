"""Importing all the required libraries """
import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

#Sklearn libraries
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#Import the keras libraries to write the NVDIA Deep learning and CNN Model architecture 
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D

#Empty list 
input_data = [] 

"""Read the driving_log.csv file and skip the first row since it has header data"""

with open('./data/driving_log.csv') as csvfile: 
    excel = csv.reader(csvfile)
    next(excel, None) 
    for data in excel:
        input_data.append(data)

"""Split the dataset into training and test dataset with split percentage of 20(which is common) . So 80% train data and 20% validation data"""
training_data, validation_data = train_test_split(input_data,test_size=0.20)

"""Defining the generator method as suggested by udacity lectures to yeild the images and respective angles """
#In general, batch size of 32 is a good starting point as per internet study.

def generator(input_data, batch_size=32):
    total_datasize = len(input_data)

    while 1: 
        shuffle(input_data) 
        for offset in range(0, total_datasize, batch_size):
            batch_inputs = input_data[offset:offset+batch_size]

            total_images = []
            total_angles = []

            for batch_input in batch_inputs:
                """Since three images ( center, left, right) , so range is 0 to 3 """
                for j in range(0,3):
                    image_name = './data/IMG/'+batch_input[j].split('/')[-1]
                    middle_image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
                    middle_angle = float(batch_input[3])
                    total_images.append(middle_image)

                    """ Add correction factor"""
                    if(j==0):
                        total_angles.append(middle_angle)
                    elif(j==1):
                        total_angles.append(middle_angle+0.15)
                    elif(j==2):
                        total_angles.append(middle_angle-0.15)
                    
                    """Since mostly we have left turns on track , need to augument the images to generate more dataset"""              
                    total_images.append(cv2.flip(middle_image,1))
                    if(j==0):
                        total_angles.append(middle_angle*-1)
                    elif(j==1):
                        total_angles.append((middle_angle+0.15)*-1)
                    elif(j==2):
                        total_angles.append((middle_angle-0.15)*-1)                   

            X = np.array(total_images)
            y = np.array(total_angles)

            yield sklearn.utils.shuffle(X, y)

"""We need to train the model using generator function """
corrected_training_data = generator(training_data, batch_size=32)
corrected_validation_data = generator(validation_data, batch_size=32)

#MODEL ARCHITECTURE
"""Use sequential model to start implementing the algorithm"""
model = Sequential()

"""Data preprocessing"""
input_shape = (160, 320, 3)
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape= input_shape))

"""Remove unnecessory part of image at top and bottom pixels"""
model.add(Cropping2D(cropping=((70,25),(0,0)))) 

"""5x5 Convolutional layers with stride of 2x2"""

#Filter depth = 24 , size = (5,5) and activation as ELU 
model.add(Conv2D(24,(5,5),strides=[2, 2]))
model.add(Activation('elu'))

#Filter depth = 36 , size = (5,5) and activation as ELU 
model.add(Conv2D(36,(5,5),strides=[2, 2]))
model.add(Activation('elu'))

#Filter depth = 48 , size = (5,5) and activation as ELU 
model.add(Conv2D(48,(5,5),strides=[2, 2]))
model.add(Activation('elu'))

"""TWO 3x3 Convolutional layers with stride of 1x1"""

#Filter depth = 64 , size = (3,3) and activation as ELU 
model.add(Conv2D(64,(3,3)))
model.add(Activation('elu'))

model.add(Conv2D(64,(3,3)))
model.add(Activation('elu'))

"""Flatten before passing to Fully Connected layers"""
model.add(Flatten())

"""Three Fully Connected Layers """
model.add(Dense(100))
model.add(Activation('elu'))

model.add(Dropout(0.25))

model.add(Dense(50))
model.add(Activation('elu'))

model.add(Dense(10))
model.add(Activation('elu'))

""" Output layer with tanh activation """
model.add(Dense(1, activation='tanh', name='output'))

"""Using adam optimizer and mse loss function"""
model.compile(optimizer="adam", loss="mse")


"""Passing the images from generator to fit function."""
model.fit_generator(corrected_training_data, samples_per_epoch= len(training_data), validation_data=corrected_validation_data,nb_val_samples=len(validation_data), nb_epoch=2, verbose=1)

model.save('model.h5')

"""Plot the summary of model """

model.summary()
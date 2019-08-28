# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:50:22 2019

@author: Eric
"""



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlite3
import keras
import keras.backend as K
from sklearn.preprocessing import normalize
from eplusparser.eplusparser import parse
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from math import sqrt
def create_dataset(dataset,names,scaler,look_back=1):
    dataX, dataY = [], []
    
    for i in range(3,len(names)):
        if "Setpoint" in names[i][1] or "Schedule" in names[i][1] or "Occupant" in names[i][1] or "Outdoor" in names[i][1]:
            dataX.append(dataset[1:,i])
        
        else:
            dataX.append(dataset[:len(dataset)-1,i])
    dataY.append(dataset[1:,1])
    dataY.append(dataset[1:,2])
    dataX=np.array(list(map(list,zip(*dataX))))
    dataX=scaler.fit_transform(dataX)
    dataX=dataX[:,np.newaxis,:]
    dataY=np.array(list(map(list,zip(*dataY))))
    dataY=scaler.fit_transform(dataY)
    return dataX, dataY

def create_model(trainX,trainY,look_back=1):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(36,input_shape=(look_back,trainX.shape[2]),activation="sigmoid",return_sequences=True))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.LSTM(36,input_shape=(look_back,trainX.shape[2]),activation="sigmoid",return_sequences=True))
    model.add(keras.layers.LSTM(18,input_shape=(look_back,trainX.shape[2]),activation="sigmoid",return_sequences=True))
    model.add(keras.layers.LSTM(trainY.shape[1], input_shape=(look_back, trainX.shape[2]),activation="sigmoid"))
    #model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
    return model

df = parse('D:\\UT Courses\\EE 364D\\Office\\run/eplusout.sql')
df.head()
dataset=df.values
names=df.columns.values.tolist()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

look_back = 1
trainX, trainY = create_dataset(dataset, names,scaler, look_back)

df1 = parse('D:\\UT Courses\\EE 364D\\Office1\\run/eplusout.sql')
df1.head()
testset = df1.values

names1=df1.columns.values.tolist()
testX, testY = create_dataset(testset, names1,scaler, look_back)

model=create_model(trainX,trainY)
model.fit(trainX, trainY, epochs=100, batch_size=10)




score, acc = model.evaluate(testX, testY, batch_size=10)
print('Test accuracy:', acc)
testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY=scaler.inverse_transform(testY)
print(testPredict)
print(testY)
rmse = sqrt(mean_squared_error(testPredict,testY))
print(rmse)

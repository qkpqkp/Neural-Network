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
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



df = parse('D:\\UT Courses\\EE 364D\\Office\\run/eplusout.sql')
df1 = parse('D:\\UT Courses\\EE 364D\\Office1\\run/eplusout.sql')
df.head()
df1.head()
print(df.shape)
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
normalized_df = normalized_df.dropna(axis='columns')
#normalized_df = normalized_df.loc[:, (normalized_df != 0).any(axis=0)]
df=normalized_df
names=df.columns.values.tolist()
y=pd.DataFrame()
x=pd.DataFrame()
for i in range(3,len(names)):
    if "Setpoint" in names[i][1] or "Schedule" in names[i][1] or "Occupant" in names[i][1] or "Outdoor" in names[i][1]:
        x=x.append(df.iloc[:,i],ignore_index=True)
    else:
        y=y.append(df.iloc[:,i],ignore_index=True)

names1=df1.columns.values.tolist()
x_test=pd.DataFrame()
y_test=pd.DataFrame()

for i in range(3,len(names1)):
    if "Setpoint" in names1[i][1] or "Schedule" in names1[i][1] or "Occupant" in names1[i][1] or "Outdoor" in names1[i][1]:
        x_test=x_test.append(df1.iloc[:,i],ignore_index=True)
    else:
        y_test=y_test.append(df1.iloc[:,i],ignore_index=True)

x=x.transpose().values
x=x[:,np.newaxis,:]
y=y.transpose().values
batch_size=40
epochs=200
print(x.shape)
print(y.shape)

x_test=x_test.transpose().values
x_test=x_test[:,np.newaxis,:]
y_test=y_test.transpose().values

def create_rnn_model():
    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(y.shape[1],input_shape=(1,x.shape[2]),activation="relu"))
    #model.add(keras.layers.SimpleRNN(y.shape[1]))
    
    model.set_weights([np.random.rand(*w.shape)*0.2 - 0.1 for w in model.get_weights()])
    return model


model_RNN = create_rnn_model()
rmsprop=keras.optimizers.rmsprop(lr=0.001)
model_RNN.compile(optimizer=rmsprop, loss='mean_squared_error', metrics=['accuracy'])
model_RNN.fit(x,y,batch_size=batch_size,epochs=epochs)

score, acc = model_RNN.evaluate(x_test, y_test, batch_size=batch_size)

print('Test accuracy:', acc)

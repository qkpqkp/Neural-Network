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
from sklearn.preprocessing import StandardScaler

df = parse('D:\\UT Courses\\EE 364D\\Office1\\run/eplusout.sql')
df.head()
print(df.shape)


names=df.columns.values.tolist()
y=pd.DataFrame()
x=pd.DataFrame()
for i in range(2,len(names)):
    if "Setpoint" in names[i][1]:
        x=x.append(df.iloc[:,i],ignore_index=True)
    else:
        y=y.append(df.iloc[:,i],ignore_index=True)


x=x.transpose().values
x=x[:,np.newaxis,:]
y=y.transpose().values
batch_size=40
epochs=1000
print(x.shape)
print(y.shape)

def create_rnn_model():
    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(y.shape[1],input_shape=(1,x.shape[2],),activation="relu",dropout=0.5,recurrent_dropout=0.5))
    model.add(keras.layers.Dense(y.shape[1]))
    model.set_weights([np.random.rand(*w.shape)*0.2 - 0.1 for w in model.get_weights()])
    return model


model_RNN = create_rnn_model()
model_RNN.compile(optimizer='sgd', loss='mean_squared_error')
model_RNN.fit(x,y,batch_size=batch_size,epochs=epochs)

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:09:54 2018

@author: Sarath.Sahadevan
"""


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


Data = [[[(i+j)/100]for i in range(5)]for j in range (100)]

target = [((i+5)/100)for i in range(100)]


data = np.array(Data,dtype = float)
target = np.array(target,dtype = float)

data.shape

target.shape

x_train, x_test, y_train, y_test = train_test_split(data,target,test_size = 0.20,train_size =0.80,random_state = 4)


#Model

model = Sequential()

model.add(LSTM((1),batch_input_shape = (None,None,1),return_sequences = True))


model.add(LSTM((4),return_sequences=True))

model.add(Dense(64, input_dim=5))

model.add(Dense(32, input_dim=5))


model.add(LSTM((1),return_sequences=False))


model.compile(loss = 'mean_absolute_error',optimizer ='adam',metrics = ['accuracy'])


#can verify the shape and the numbe rof parameter

model.summary()


history = model.fit(x_train,y_train,epochs=400,validation_data=(x_test,y_test))


results = model.predict(x_test)



plt.scatter(range(20),results,c='r')

plt.scatter(range(20),y_test,c='g')

plt.show()

plt.plot(history.history['loss'])

plt.show()




















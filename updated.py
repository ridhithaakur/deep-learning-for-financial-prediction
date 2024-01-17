#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 23:10:36 2018

@author: vyomunadkat
"""

#iTd3kPLBcnsLKxy5Wsxg


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl
from skimage.measure import compare_ssim as ssim
from scipy import spatial


# add quandl API key for unrestricted
quandl.ApiConfig.api_key = 'iTd3kPLBcnsLKxy5Wsxg'

# get the table for daily stock prices and,
# filter the table for selected tickers, columns within a time range
# set paginate to True because Quandl limits tables API to 10,000 rows per call
data = quandl.get_table('WIKI/PRICES', ticker = ['FB'], 
                        qopts = { 'columns': ['ticker', 'date', 'open'] }, 
                        date = { 'gte': '2012-01-01', 'lte': '2017-06-30' }, 
                        paginate=True)

# create a new dataframe with 'date' column as index
new = data.set_index('date')

# use pandas pivot function to sort adj_close by tickers
clean_data = new.pivot(columns='ticker')





#importing the dataset

training_set = clean_data

#feature scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

training_set_scaled = scaler.fit_transform(training_set)

#creating data with 20 timestamps
# Creating a data structure with 20 timesteps and t+1 output
X_train = []
y_train = []
for i in range(20, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-20:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


"""
#getting the data to train

train_set = training_set[0:1257]
test_set = training_set[1:1258]


#reshape

train_set = np.reshape(train_set, (1257,1,1))"""

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#import keras libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#initialising the model

regressor = Sequential()

#adding layers

#1st layer
regressor.add(LSTM(units=64, return_sequences = True, input_shape = (None,1)))
#in input shape, none signifies any number of timestamp and 1 is the no. of feature i.e. opening price in this case

#2nd layer
regressor.add(LSTM(units=64, return_sequences = True))

#3rd layer
regressor.add(LSTM(units=64, return_sequences = True))

#4th layer
regressor.add(LSTM(units=64))



regressor.add(Dense(units = 1))

#compiling
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')

#fitting the training set

regressor.fit(X_train, y_train, batch_size=32, epochs=200)

#getting the actual price to be predicted

# get the table for daily stock prices and,
# filter the table for selected tickers, columns within a time range
# set paginate to True because Quandl limits tables API to 10,000 rows per call
data = quandl.get_table('WIKI/PRICES', ticker = ['FB'], 
                        qopts = { 'columns': ['ticker', 'date', 'open'] }, 
                        date = { 'gte': '2017-07-01', 'lte': '2018-08-30' }, 
                        paginate=True)

# create a new dataframe with 'date' column as index
new = data.set_index('date')

# use pandas pivot function to sort adj_close by tickers
clean_data = new.pivot(columns='ticker')






#actual_price = pd.read_csv('./Google_Stock_Price_Test.csv')

#actual_price = actual_price.iloc[:,1:2].values
actual_price = clean_data
real_stock_price = np.concatenate((training_set[0:len(training_set_scaled)], actual_price), axis = 0)


# Getting the predicted stock price of 2017
scaled_real_stock_price = scaler.fit_transform(real_stock_price)
inputs = []
for i in range(len(training_set_scaled), len(real_stock_price)):
    inputs.append(scaled_real_stock_price[i-20:i, 0])
inputs = np.array(inputs)
test_input = scaler.inverse_transform(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualising the results
#plt.plot(real_stock_price[len(training_set_scaled):], color = 'red', label = 'Real Google Stock Price')
plt.plot(np.array(actual_price), color = 'red', label = 'Real Adobe Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Adobe Stock Price')
plt.title('Adobe Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Adobe Stock Price')
plt.legend()
plt.show()

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def similar(imageA, imageB):
    m = mse(imageA, imageB)
    s = 1 - spatial.distance.cosine(imageA, imageB)
    b = spatial.distance.euclidean(imageA, imageB)
    print(m)
    print(s)
    if (s>=0.97 and m<=30):
        return True
    else:
        return False

similar(np.array(actual_price),predicted_stock_price)

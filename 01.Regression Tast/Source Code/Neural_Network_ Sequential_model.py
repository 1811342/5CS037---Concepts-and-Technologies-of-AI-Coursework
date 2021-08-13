# model and pre_pro
import tensorflow as tf  # tenser flow
from tensorflow import keras  # keras
from keras.models import Sequential  # sequential layer
from keras.layers import Activation, Dense  # dense layer
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.model_selection import train_test_split  # data split
from sklearn.preprocessing import MinMaxScaler  # normalisation
from keras.utils.vis_utils import plot_model

# for plotting
import matplotlib.pyplot as plt
import math

# data reading and arrays
import pandas as pd
from sklearn.datasets import load_boston
import numpy as np

# load data indto a pandas data frame
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

# adding the price column to the dataframe
# data['PRICE'] = boston.target
# dataset = data.values

# create lables and samples and adding values
train_labels = []
train_samples = []
train_samples = data.values
train_labels = boston.target

# transfer data to np arrrays
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples)

#print(scaled_train_samples)

# split data train test and validation
x_train, x_test, y_train, y_test = train_test_split(scaled_train_samples, train_labels, test_size=0.2)

"""
# data shapes
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)
"""

model = Sequential()

model.add(Dense(252, input_shape=(13,), activation='relu'))  # input layer and first dense layer
model.add(Dense(128, activation='relu'))  # hidden second Dense layer
model.add(Dense(1, activation='linear'))  # output layer and third Dense layer

# model layers graph
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# model details
# model.summary()

# fit model and compile adam optimizer and mse and mae
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history=model.fit(x_train, y_train, epochs=100, validation_split=0.05)

# these graphs are from keras website codes.
# graph model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# graph model mae
plt.plot(history.history['mae']) # plot mae of train data
plt.plot(history.history['val_mae']) # plot mae of validation data
plt.title('Model Mean Absolute Error')
# X y labels
plt.ylabel('mean absolute error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


# predict using the model
predictions = model.predict(x_test)

# display predictions
for i in range(len(predictions)):
    print("Actual: ", y_test[i], " Prediction: ", predictions[i])

"""
true_handle = plt.scatter(y_test, y_test, alpha=0.6, color='blue', label='value')
fit = np.poly1d(np.polyfit(y_test,y_test,1))
lims = np.linspace(min(y_test) - 1, max(y_test) + 1)
plt.plot(lims, fit(lims), alpha=0.3, color='black')
pred_handle = plt.scatter(y_test, predictions, alpha=0.6, color='red', label='predicted')
plt.legend(handles=[true_handle,pred_handle], loc='upper left')
plt.show()
"""




















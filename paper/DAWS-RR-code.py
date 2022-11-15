# -*- coding: utf-8 -*-
"""
Created on 2022
@author: Hsin-Yuan Chang, hy.chang@m109.nthu.edu.tw
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Multiply, Lambda, Reshape, MaxPool2D
from matplotlib import pyplot as plt
import scipy.io
import math

##############   Parameter settings   ##########################
    
num_data_train= 175*22
num_data_test= 175*10
ObsTime= 6
slow_Ts= 100
data_length= ObsTime*slow_Ts

##############   Load dataset   ################################
print("start loading")
X_train = scipy.io.loadmat('./DAWS_RR_data_bin_breath.mat')
# X_train is of size [num_data_train, number_of_bin, data_length, 3]
# The last "3" indicates the number of features we use: magnitude, phase of complex signal, frequency response of vital sign
# number_of_bin is set to 5 in our study
est_train = scipy.io.loadmat('./DAWS_RR_data_est_breath.mat')
# est_train is of size [num_data_train, number_of_frequencies, number_of_bin]
# The number_of_frequencies is generally set to "2": respiration rate and heart rate
y_train = scipy.io.loadmat('./DAWS_RR_data_ref_breath.mat')
# y_train is of size [num_data_train, number_of_frequencies]
# y_train is the label of respiration rate and heart rate
# Note that we only adopt the heart rate part in this code
X_test = scipy.io.loadmat('./DAWS_RR_data_bin_test_breath.mat')
# X_test is of size [num_data_test, number_of_bin, data_length, 3]
est_test = scipy.io.loadmat('./DAWS_RR_data_est_test_breath.mat')
# est_test is of size [num_data_test, number_of_frequencies, number_of_bin]
y_test = scipy.io.loadmat('./DAWS_RR_data_ref_test_breath.mat')
# y_test is of size [num_data_test, number_of_frequencies]
print("complete loading")

X_train = X_train["X_train"]
est_train = est_train["est_train"]
y_train = y_train["y_train"]
X_test = X_test["X_test"]
est_test = est_test["est_test"]
y_test = y_test["y_test"]




def freq_func(temp):
    f_est,f_weight = temp
    f_weight = f_weight[:,:,tf.newaxis]
    f_vec=f_weight[:,:,0,0]*f_est[:,:,0]
    freq = tf.reduce_sum(f_vec,axis=1)

    return freq


bins = tf.keras.Input(name='bins', shape=(5,data_length,3))
f_est = tf.keras.Input(name='f_est', shape=(5,1))
temp = tf.keras.layers.Conv2D(5, (5,50), activation="tanh",padding = 'same')(bins)
temp = tf.keras.layers.Conv2D(10, (5,50), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 3))(temp)
temp = tf.keras.layers.Conv2D(10, (5,5), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 3))(temp)
temp = tf.keras.layers.Flatten()(temp)

f_weight = Dense(5, name='f_weight')(temp)
f_weight_r = Reshape((5, 1),name='f_weight_r')(f_weight)
freq = Lambda(freq_func, dtype=tf.float32, output_shape=(1,),name='freq')([f_est, f_weight_r])

model = tf.keras.models.Model(inputs=[bins,f_est], outputs=[f_weight_r,freq])
model.summary()  # show output shape and parameters of the payers
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer='adam', 
              loss={'f_weight_r' :'mse',
                    'freq' :'mse',}
              ,loss_weights=[0.0,1.0])

train_history = model.fit(x=[X_train,est_train[:,:,0]], y=[np.zeros([num_data_train,5,1]),y_train[:,0]], validation_split=0.2, epochs=50, batch_size=64, shuffle=True)

#plot Loss Curve
training_loss=train_history.history['loss']
val_loss=train_history.history['val_loss']
plt.figure(figsize=(6,4))
plt.plot(training_loss,label="training_loss")
plt.plot(val_loss,label="validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend(loc="best")
plt.show

fweight_res, pred_res = model.predict([X_test, est_test[:,:,0]])
predict_err=np.mean((pred_res-y_test[:,0])**2)
print(predict_err)


scipy.io.savemat("./DAWS_RR_results.mat", {"DAWS_RR_error":pred_res-y_test[:,0]})


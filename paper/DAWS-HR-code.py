# -*- coding: utf-8 -*-
"""
Created on 2022
@author: Hsin-Yuan Chang, hy.chang@m109.nthu.edu.tw
"""


import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Multiply, Lambda, Reshape, MaxPool2D
from matplotlib import pyplot as plt
import scipy as sp
import scipy.io
import math
##############   Parameter settings   ##########################

num_data_train= 175*16
num_data_test= 175*10
ObsTime= 6
slow_Ts= 100
data_length= ObsTime*slow_Ts

##############   Load dataset   ################################
print("start loading")
X_train = scipy.io.loadmat('./DAWS_HR_data_bin.mat')
# X_train is of size [num_data_train, number_of_bin, data_length, 3]
# The last "3" indicates the number of features we use: magnitude, phase of complex signal, frequency response of vital sign
# number_of_bin is set to 5 in our study
est_train = scipy.io.loadmat('./DAWS_HR_data_est.mat')
# est_train is of size [num_data_train, number_of_frequencies, number_of_bin]
# The number_of_frequencies is generally set to "2": respiration rate and heart rate
y_train = scipy.io.loadmat('./DAWS_HR_data_ref.mat')
# y_train is of size [num_data_train, number_of_frequencies]
# y_train is the label of respiration rate and heart rate
# Note that we only adopt the heart rate part in this code
X_test = scipy.io.loadmat('./DAWS_HR_data_bin_test.mat')
# X_test is of size [num_data_test, data_length, number_of_bin, 3]
est_test = scipy.io.loadmat('./DAWS_HR_data_est_test.mat')
# est_test is of size [num_data_test, number_of_frequencies, number_of_bin]
y_test = scipy.io.loadmat('./DAWS_HR_data_ref_test.mat')
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
    f_vec=f_weight[:,:,0]*f_est[:,:,0]
    freq = tf.reduce_sum(f_vec,axis=1)

    return freq


bins = tf.keras.Input(name='bins', shape=(5,data_length,3))
f_est = tf.keras.Input(name='f_est', shape=(5,1))
temp = tf.keras.layers.Conv2D(3, (5,100), activation="selu",padding = 'same')(bins)
temp = tf.keras.layers.Conv2D(3, (5,50), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.Conv2D(128, (3,10), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
temp = tf.keras.layers.Conv2D(128, (5,10), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
temp = tf.keras.layers.Conv2D(3, (5,80), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
net1 = tf.keras.models.Model(inputs = [bins, f_est], outputs = [temp])
net1.summary()
feature_layer_size = 75

net2_input = tf.keras.Input(name='features', shape=(5,feature_layer_size,3))
net2_f = tf.keras.Input(name='f_est', shape=(5,1))
temp = tf.keras.layers.Conv2D(5, (5,50), activation="tanh",padding = 'same')(net2_input)
temp = tf.keras.layers.Conv2D(5, (5,50), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
temp = tf.keras.layers.Conv2D(5, (5,100), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.Conv2D(5, (5,100), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
temp = tf.keras.layers.Conv2D(5, (5,100), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.Conv2D(5, (5,100), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)


temp = tf.keras.layers.Flatten()(temp)
f_weight = Dense(5, name='f_weight',activation=tf.keras.activations.selu)(temp)
f_weight_hr = Reshape((5, 1),name = 'f_weight_hr')(f_weight)
freq = Lambda(freq_func, dtype=tf.float32, output_shape=(1,),name='freq')([net2_f,f_weight_hr])
net2 = tf.keras.models.Model(inputs = [net2_input, net2_f], outputs = [f_weight_hr,freq], name='net2')

full_bins = tf.keras.Input(name='full_bins', shape=(5,data_length,3))
full_f_est = tf.keras.Input(name='full_f_est', shape=(5,1))
full_feature = net1([full_bins, full_f_est])
full_f_weight_hr, full_freq = net2([full_feature, full_f_est])

model = tf.keras.models.Model(inputs=[full_bins, full_f_est], outputs=[full_f_weight_hr,full_freq])
model.summary()  # show output shape and parameters of the payers
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer='adam', 
              loss={'net2' :'mse',
                    'net2_1' :'mse',}
              ,loss_weights=[0.0,1.0])

train_history = model.fit(x=[X_train,est_train[:,:,1]], y=[np.zeros([num_data_train,5,1]),y_train[:,1]], validation_split=0.2, epochs=30, batch_size=64, shuffle=False)

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
plt.show()


    
fweight_heart, pred_heart = model.predict([X_test,est_test[:,:,1]])
predict_err=np.mean((pred_heart-y_test[:,1])**2)
print(predict_err)


scipy.io.savemat("./DAWS_HR_results.mat", {"DAWS_HR_error":pred_heart-y_test[:,1]})
    

"""
#################################################################################################
"""


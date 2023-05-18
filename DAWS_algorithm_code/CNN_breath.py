# -*- coding: utf-8 -*-
"""
Created on 2023/03/25

CNN model of breathing

The model takes the feature of range profile matrix as input and generates weights for each range bin.
Finally, data fusion is performed based on these weights.
"""
import tensorflow as tf
import numpy as np
#from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Multiply, Lambda, Reshape, MaxPool2D
from matplotlib import pyplot as plt
#import mat4py
#from mat4py import loadmat
import scipy.io
# from keras.layers.advanced_activations import PReLU
import math
import time


" set up ---------------------------------------------------------------------------=------------------- "
execution_numbering_str = '01'

epochs_num = 30

data_time_length = 175     # 175 = 180 - 6 + 1 (180 sec signal sliding with 6 sec window, each sliding step is 1 sec)
data_num_train = 22
data_num_test = 10

size_data_train = data_time_length * data_num_train 
size_data_test = data_time_length * data_num_test
ObsTime = 6           # each observation window is 6 sec
slow_time_fre = 100   # 100 chirp in 1 sec
input_bin_length = slow_time_fre * ObsTime
" ------------------------------------------------------------------------------------------------------ "

" loading ---------------------------------------------------------------------------------------------- "
print("---\nstart loading")
# rawx_train = scipy.io.loadmat('./../data_for_model_input/training/data_bin_train.mat')
rawx_train = scipy.io.loadmat('./data/data_for_model_input/breathing/data_feature_bin_train.mat')
estFre_train = scipy.io.loadmat('./data/data_for_model_input/breathing/data_esti_freq_train.mat')
rawy_train = scipy.io.loadmat('./data/data_for_model_input/breathing/data_ref_train.mat')
rawx_test = scipy.io.loadmat('./data/data_for_model_input/breathing/data_feature_bin_test.mat')
estFre_test = scipy.io.loadmat('./data/data_for_model_input/breathing/data_esti_freq_test.mat')
rawy_test = scipy.io.loadmat('./data/data_for_model_input/breathing/data_ref_test.mat')
print("complete loading\n---")
" ------------------------------------------------------------------------------------------------------ "

" data ------------------------------------------------------------------------------------------------- "
# Convert string values from a dictionary into int datatypes
X_train_trans=np.zeros([size_data_train,ObsTime*slow_time_fre,5,3])
for dicts in rawx_train:
    if dicts=='data_bin':
        for index in range(len(rawx_train[dicts])):
            X_train_trans[index] = rawx_train[dicts][index]
y_train=np.zeros([size_data_train,2])
for dicts in rawy_train:
    if dicts=='data_ref':
        for index in range(len(rawy_train[dicts])):
            y_train[index] = rawy_train[dicts][index]
y_test=np.zeros([size_data_test,2])
for dicts in rawy_test:
    if dicts=='data_ref':
        for index in range(len(rawy_test[dicts])):
            y_test[index] = rawy_test[dicts][index]
est_train_trans=np.zeros([size_data_train,2,5])
for dicts in estFre_train:
    if dicts=='data_est':
        for index in range(len(estFre_train[dicts])):
            est_train_trans[index] = estFre_train[dicts][index]
X_test_trans=np.zeros([size_data_test,ObsTime*slow_time_fre,5,3])
for dicts in rawx_test:
    if dicts=='data_bin':
        for index in range(len(rawx_test[dicts])):
            X_test_trans[index] = rawx_test[dicts][index]
est_test_trans=np.zeros([size_data_test,2,5])
for dicts in estFre_test:
    if dicts=='data_est':
        for index in range(len(estFre_test[dicts])):
            est_test_trans[index] = estFre_test[dicts][index]

X_train = X_train_trans.transpose((0, 2, 1, 3)) # data_bin_train (contains amplitude, phase, and frequency response of phase of range profile)
X_test = X_test_trans.transpose((0, 2, 1, 3))
est_train = est_train_trans.transpose((0, 2, 1)) # esti-freq
est_test = est_test_trans.transpose((0, 2, 1))

del rawx_train, rawy_train, rawx_test, estFre_train, estFre_test
del X_train_trans, X_test_trans, est_train_trans, est_test_trans
" ------------------------------------------------------------------------------------------------------ "

" custom layer of network (do data fusion by predicted weight) ----------------------------------------- "
def freq_fusion_layer(temp):                 # sum by weight (freq_est_bin_1*f_weight_1 + freq_est_bin_2*f_weight_2 + ...)
    freq_est_bin,f_weight = temp
    
    # f_weight = tf.nn.softmax(f_weight)
    # print(f_weight)
    # %debug print(f_weight)
    f_weight = f_weight[:,:,tf.newaxis]
    # f_res_vec=f_weight[:,:,0]*freq_est_bin[:,:,0]
    # f_heart_vec=f_weight[:,:,1]*freq_est_bin[:,:,1]
    # f_res = tf.reduce_sum(f_res_vec,axis=1)
    # f_heart = tf.reduce_sum(f_heart_vec,axis=1)
    # freq_est_final = tf.stack([f_res,f_heart],axis = 1)
    f_vec = f_weight[:,:,0]*freq_est_bin[:,:,0]
    # print(f_vec)
    # %debug print(f_vec)
    # ipdb.set_trace(context=6)
    freq_est_final = tf.reduce_sum(f_vec,axis=1)

    return freq_est_final
" ------------------------------------------------------------------------------------------------------ "

" network structure (functional API) ------------------------------------------------------------------- "
freq_est_bin = tf.keras.Input(name='freq_est_bin', shape=(5,1))                 # estimated frequency of each features_bin

features_bin = tf.keras.Input(name='features_bin', shape=(5,input_bin_length,3))       # features of each bin from range profile matrix
temp = tf.keras.layers.Conv2D(5, (5,50), activation="sigmoid",padding = 'same')(features_bin)
temp = tf.keras.layers.Conv2D(10, (5,50), activation="sigmoid",padding = 'same')(temp)
temp = tf.keras.layers.BatchNormalization(axis=1)(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 3))(temp)
temp = tf.keras.layers.Conv2D(10, (5,5),padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 3))(temp)
temp = tf.keras.layers.Flatten()(temp)
weight_pred_bin = tf.keras.layers.Dense(5, activation="sigmoid", name='weight_pred_bin')(temp)


freq_est_final = tf.keras.layers.Lambda(freq_fusion_layer, dtype=tf.float32, output_shape=(1,), name='freq_est_final')([freq_est_bin,weight_pred_bin])
" ------------------------------------------------------------------------------------------------------ "

" model ------------------------------------------------------------------------------------------------ "
model = tf.keras.models.Model(inputs=[features_bin,freq_est_bin], outputs=[weight_pred_bin,freq_est_final])
model.summary()  # show output shape and parameters of the payers
# 編譯(compile): 選擇損失函數、優化方法及成效衡量方式
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer='adam', 
              loss={'weight_pred_bin' :'mse',
                    'freq_est_final' :'mse',}
              ,loss_weights=[0.0,1.0],run_eagerly=False)
" ------------------------------------------------------------------------------------------------------ "

" training --------------------------------------------------------------------------------------------- "
print("---\nstart training")
time_start = time.time()  # calculate the execution time
train_history = model.fit(x=[X_train,est_train[:,:,0]], y=[np.zeros([size_data_train,5,1]),y_train[:,0]], validation_split=0.2, epochs=epochs_num, batch_size=64, shuffle=True)
time_end = time.time()
print("complete training\n---")
print(round(time_end - time_start), "seconds, i.e.", round((time_end - time_start)/60/60,2), "hours")

training_loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
loss_fig = plt.figure(figsize = (6,4))
plt.plot(training_loss,label = "training_loss")
plt.plot(val_loss,label = "validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend(loc="best")
plt.show
" ------------------------------------------------------------------------------------------------------ "

" output ----------------------------------------------------------------------------------------------- "

weight_pred_breath, freq_pred_breath = model.predict([X_test,est_test[:,:,0]])
# np.save('./data/model_output/weight_h.npy', weight_pred_breath)
# np.save('./data/model_output/freq_h.npy', freq_pred_breath)
# mat = np.load(' .npy')
# scipy.io.savemat(' .mat', {'light_light': mat})

scipy.io.savemat('./data/model_output/test_weigth.mat', {'weight_predicted': weight_pred_breath})
scipy.io.savemat('./data/model_output/test_freq.mat', {'freq_predicted': freq_pred_breath})
" ------------------------------------------------------------------------------------------------------ "












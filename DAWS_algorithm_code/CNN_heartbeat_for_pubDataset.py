# -*- coding: utf-8 -*-
"""
Created on 2023/03/25

CNN model of heartbeat

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
import h5py


" set up ----------------------------------------------------------------------------------------------- "
execution_numbering_str = '01'
FFT_or_ZFFT = 1                # 0: FFT  1: ZFFT

zoom_factor_ZFFT = 4

selected_bin_num_default = 5       # default: select 5 range bins
data_time_length = 171     # 175 = 180 - 6 + 1 (180 sec signal sliding with 6 sec window, each sliding step is 1 sec)
data_num_train = 15
data_num_test = 10

if FFT_or_ZFFT == 0:
    str_FFT_or_ZFFT = 'FFT'
    zoom_factor = 1
    epochs_num = 50
    batch_size_num = 64
    
    kernel_size_height = selected_bin_num_default
    opt = tf.keras.optimizers.Adam()
    
    learning_rate_value = 0.001
    # batch_size_num = 32
    # opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)
    
    
elif FFT_or_ZFFT == 1:
    str_FFT_or_ZFFT = 'ZFFT'
    zoom_factor = zoom_factor_ZFFT
    epochs_num = 100
    batch_size_num = 4
    learning_rate_value = 0.0001
    kernel_size_height = selected_bin_num_default * zoom_factor
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_value)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)
    
    # batch_size_num = 4
    # learning_rate_value = 0.0001/500
    # batch_size_num = 4 * 8
    # learning_rate_value = 0.0001/100 * 8
" ------------------------------------------------------------------------------------------------------ "


" parameter -------------------------------------------------------------------------------------------- "
selected_bin_num = zoom_factor * selected_bin_num_default
size_data_train = data_time_length * data_num_train 
size_data_test = data_time_length * data_num_test
ObsTime = 10           # each observation window is 6 sec
slow_time_fre = 20   # 100 chirp in 1 sec
input_bin_length = slow_time_fre * ObsTime
" ------------------------------------------------------------------------------------------------------ "

" loading ---------------------------------------------------------------------------------------------- "
print("---\nstart loading")
# rawx_train = scipy.io.loadmat('./for public dataset/data_for_model_input/' +str_FFT_or_ZFFT+ '/data_feature_bin_train.mat')
estFre_train = scipy.io.loadmat('./for public dataset/data_for_model_input/' +str_FFT_or_ZFFT+ '/data_esti_freq_train.mat')
rawy_train = scipy.io.loadmat('./for public dataset/data_for_model_input/' +str_FFT_or_ZFFT+ '/data_ref_train.mat')
# rawx_test = scipy.io.loadmat('./for public dataset/data_for_model_input/' +str_FFT_or_ZFFT+ '/data_feature_bin_test.mat')
estFre_test = scipy.io.loadmat('./for public dataset/data_for_model_input/' +str_FFT_or_ZFFT+ '/data_esti_freq_test.mat')
rawy_test = scipy.io.loadmat('./for public dataset/data_for_model_input/' +str_FFT_or_ZFFT+ '/data_ref_test.mat')

if FFT_or_ZFFT == 0:
    rawx_train = scipy.io.loadmat('./for public dataset/data_for_model_input/' +str_FFT_or_ZFFT+ '/data_feature_bin_train.mat')
    rawx_test = scipy.io.loadmat('./for public dataset/data_for_model_input/' +str_FFT_or_ZFFT+ '/data_feature_bin_test.mat')
elif FFT_or_ZFFT == 1:
    """ for matlab mat files is v7.3 (MAT-file version 7.3) """
    training_feature_bin = h5py.File('./for public dataset/data_for_model_input/ZFFT/data_feature_bin_train.mat',mode='r')
    data_temp = training_feature_bin['data_bin_ZFFT'][:]
    data_temp = np.transpose(data_temp)
    rawx_train = dict()
    rawx_train['data_bin_ZFFT'] = data_temp
    training_feature_bin.close()
    testing_feature_bin = h5py.File('./for public dataset/data_for_model_input/ZFFT/data_feature_bin_test.mat',mode='r')
    data_temp = testing_feature_bin['data_bin_ZFFT'][:]
    data_temp = np.transpose(data_temp)
    rawx_test = dict()
    rawx_test['data_bin_ZFFT'] = data_temp
    testing_feature_bin.close()

" data ------------------------------------------------------------------------------------------------- "
# Convert string values from a dictionary into int datatypes
X_train_trans=np.zeros([size_data_train,ObsTime*slow_time_fre,selected_bin_num,3], dtype=np.float16)
for dicts in rawx_train:
    if dicts=='data_bin_' + str_FFT_or_ZFFT:
        for index in range(len(rawx_train[dicts])):
            X_train_trans[index] = rawx_train[dicts][index]
y_train=np.zeros([size_data_train,2])
for dicts in rawy_train:
    if dicts=='data_ref_' + str_FFT_or_ZFFT:
        for index in range(len(rawy_train[dicts])):
            y_train[index] = rawy_train[dicts][index]
y_test=np.zeros([size_data_test,2])
for dicts in rawy_test:
    if dicts=='data_ref_' + str_FFT_or_ZFFT:
        for index in range(len(rawy_test[dicts])):
            y_test[index] = rawy_test[dicts][index]
est_freq_train_trans=np.zeros([size_data_train,2,selected_bin_num])
for dicts in estFre_train:
    if dicts=='data_est_' + str_FFT_or_ZFFT:
        for index in range(len(estFre_train[dicts])):
            est_freq_train_trans[index] = estFre_train[dicts][index]
X_test_trans=np.zeros([size_data_test,ObsTime*slow_time_fre,selected_bin_num,3], dtype=np.float16)
for dicts in rawx_test:
    if dicts=='data_bin_' + str_FFT_or_ZFFT:
        for index in range(len(rawx_test[dicts])):
            X_test_trans[index] = rawx_test[dicts][index]
est_freq_test_trans=np.zeros([size_data_test,2,selected_bin_num])
for dicts in estFre_test:
    if dicts=='data_est_' + str_FFT_or_ZFFT:
        for index in range(len(estFre_test[dicts])):
            est_freq_test_trans[index] = estFre_test[dicts][index]

X_train = X_train_trans.transpose((0, 2, 1, 3)) # data_bin_train (contains amplitude, phase, and frequency response of phase of range profile)
X_test = X_test_trans.transpose((0, 2, 1, 3))
est_freq_train = est_freq_train_trans.transpose((0, 2, 1)) # esti-freq
est_freq_test = est_freq_test_trans.transpose((0, 2, 1))

del rawx_train, rawy_train, rawx_test, estFre_train, estFre_test
del X_train_trans, X_test_trans, est_freq_train_trans, est_freq_test_trans
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
freq_est_bin = tf.keras.Input(name='freq_est_bin', shape=(selected_bin_num,1))                 # estimated frequency of each features_bin

features_bin = tf.keras.Input(name='features_bin', shape=(selected_bin_num,input_bin_length,3))       # features of each bin from range profile matrix

temp = tf.keras.layers.Conv2D(3, (kernel_size_height,30), activation="selu",padding = 'same')(features_bin)
temp = tf.keras.layers.Conv2D(3, (kernel_size_height,15), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.Conv2D(128, (3*zoom_factor,10), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
temp = tf.keras.layers.Conv2D(128, (kernel_size_height,10), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
temp = tf.keras.layers.Conv2D(3, (kernel_size_height,25), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
temp = tf.keras.layers.Conv2D(5, (kernel_size_height,15), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.Conv2D(5, (kernel_size_height,15), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
temp = tf.keras.layers.Conv2D(5, (kernel_size_height,30), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.Conv2D(5, (kernel_size_height,30), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
temp = tf.keras.layers.Conv2D(5, (kernel_size_height,30), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.Conv2D(5, (kernel_size_height,30), activation="tanh",padding = 'same')(temp)
temp = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(temp)
    
temp = tf.keras.layers.Flatten()(temp)
weight_pred_bin = tf.keras.layers.Dense(selected_bin_num, activation="sigmoid", name='weight_pred_bin')(temp)


freq_est_final = tf.keras.layers.Lambda(freq_fusion_layer, dtype=tf.float32, output_shape=(1,), name='freq_est_final')([freq_est_bin,weight_pred_bin])
" ------------------------------------------------------------------------------------------------------ "

" model ------------------------------------------------------------------------------------------------ "
model = tf.keras.models.Model(inputs=[features_bin,freq_est_bin], outputs=[weight_pred_bin,freq_est_final])
model.summary()  # show output shape and parameters of the payers
tf.keras.utils.plot_model(model, to_file='./model/' + execution_numbering_str + '_model_visualization.png', show_shapes=True)
# 編譯(compile): 選擇損失函數、優化方法及成效衡量方式
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.compile(optimizer=opt, 
              loss={'weight_pred_bin' :'mse',
                    'freq_est_final' :'mse',}
              ,loss_weights=[0.0,1.0],run_eagerly=False)
" ------------------------------------------------------------------------------------------------------ "

" training --------------------------------------------------------------------------------------------- "
print("---\nstart training")

time_start = time.time()  # calculate the execution time
if 'reduce_lr' in globals():
    train_history = model.fit(x=[X_train,est_freq_train[:,:,1]], y=[np.zeros([size_data_train,selected_bin_num,1]),y_train[:,1]], \
                          validation_split=0.2, epochs=epochs_num, batch_size=batch_size_num, shuffle=True, callbacks=[reduce_lr])
else:
    train_history = model.fit(x=[X_train,est_freq_train[:,:,1]], y=[np.zeros([size_data_train,selected_bin_num,1]),y_train[:,1]], \
                          validation_split=0.2, epochs=epochs_num, batch_size=batch_size_num, shuffle=True)
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
plt.title("Loss Curve (" +str_FFT_or_ZFFT+ ", batch size = " + str(batch_size_num) + ")")
plt.legend(loc="best")
plt.show
loss_fig.savefig('./for public dataset/model/'+execution_numbering_str+'_training_loss_history_'+str_FFT_or_ZFFT+'.png')

model.save('./for public dataset/model/model_save/heartbeat/'+execution_numbering_str+'_trained_model_' + str_FFT_or_ZFFT + '_'+ str(zoom_factor) + '.h5')
# model.load_weights('./model/model_save/01_trained_model_FFT_1.h5')
" ------------------------------------------------------------------------------------------------------ "

" load model ------------------------------------------------------------------------------------------- "
# model = tf.keras.models.load_model('./model/model_save/trained_model.h5')

model = tf.keras.models.Model(inputs=[features_bin,freq_est_bin], outputs=[weight_pred_bin,freq_est_final])
model.load_weights('./for public dataset/model/model_save/heartbeat/'+execution_numbering_str+'_trained_model_' + str_FFT_or_ZFFT + '_'+ str(zoom_factor) + '.h5')
" ------------------------------------------------------------------------------------------------------ "

# model.load_weights('./model/model_save/heartbeat/01_trained_model_ZFFT_2.h5')

" output ----------------------------------------------------------------------------------------------- "
weight_pred_heartbeat, freq_pred_heartbeat = model.predict([X_test,est_freq_test[:,:,1]])

scipy.io.savemat('./for public dataset/model_output/'+str_FFT_or_ZFFT+'/test_weigth.mat', {'weight_predicted_' + str_FFT_or_ZFFT: weight_pred_heartbeat})
scipy.io.savemat('./for public dataset/model_output/'+str_FFT_or_ZFFT+'/test_freq.mat', {'freq_predicted_' + str_FFT_or_ZFFT: freq_pred_heartbeat})
" ------------------------------------------------------------------------------------------------------ "












#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:27:56 2022

python Attention_PK_MASTER.py --model_name=bulid_pk_model

python Attention_PK_MASTER.py --model_name=pk_model

@author: zhangj2
"""
# In[] Libs
import os
import numpy as np
os.getcwd()
import argparse

import datetime
import keras
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Input, Dense, Dropout, Flatten,Embedding, LSTM,GRU,Bidirectional
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,BatchNormalization,Reshape
from keras.layers import UpSampling1D,AveragePooling1D,AveragePooling2D,TimeDistributed 
from keras.layers import UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda,concatenate,add,Conv2DTranspose,Concatenate
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras.layers import Reshape

from keras_self_attention import SeqSelfAttention

from keras import backend as K
import tensorflow as tf

# In[]
# Set GPU
#==========================================# 
def start_gpu(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print('Physical GPU：', len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('Logical GPU：', len(logical_gpus))

#==========================================#
# Set Configures
#==========================================# 
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU",
                        default="3",
                        help="set gpu ids") 
    
    parser.add_argument("--model_name",
                        default="bulid_pk_model",
                        help="bulid_pk_model/pk_model")
    
    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        help="number of epochs (default: 10)")
    
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="batch size")
    
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="learning rate")
    
    parser.add_argument("--patience",
                        default=2,
                        type=int,
                        help="early stopping")
    
    parser.add_argument("--monitor",
                        default="val_loss",
                        help="monitor the val_loss/loss/acc/val_acc")  
    
    parser.add_argument("--monitor_mode",
                        default="min",
                        help="min/max/auto") 
    
    parser.add_argument("--loss",
                        default='mse',
                        help="loss fucntion")  
        
    
    args = parser.parse_args()
    return args

# In[]  A size that is an integer multiple of 8
def build_pk_model(time_input=(400,1)):
    
    inp = Input(shape=time_input, name='input')
    
    x = Conv1D(16, 5, padding = 'same', activation = 'relu')(inp)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(32, 3, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)    
    
    x = Conv1D(64, 3, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = keras.layers.LSTM(units=128, return_sequences=True)(x)
    
    at_x,wt = SeqSelfAttention(return_attention=True, attention_width= 20,  
                            attention_activation='relu',name='Atten')(x)
    #----------------------#
    x1 = UpSampling1D(2)(at_x)
    x1 = Conv1D(64, 3, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    x1 = UpSampling1D(2)(x1)
    x1 = Conv1D(32, 3, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    x1 = UpSampling1D(2)(x1)
    x1 = Conv1D(16, 3, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    out1 = Conv1D(1, 3, padding = 'same', activation = 'sigmoid',name='pk')(x1)
    
    model = Model(inp, out1)
    
    return model
# In[] suit for any input

def Conv2d_BN1(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2d_BN2(x, nb_filter, kernel_size, strides=(4,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN1(x, filters, kernel_size, strides=(4,1), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN2(x, filters, kernel_size, strides=(1,1), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN3(x, filters, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = UpSampling2D(size=(4,1))(x) #1
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x    
    
def crop_and_cut(net):
    net1,net2=net
    net1_shape = net1.get_shape().as_list()
    # net2_shape = net2.get_shape().as_list()
    offsets = [0, 0, 0, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)
    return net2_resize 

# def my_reshape(x,a,b):
#     return K.reshape(x,(-1,a,b)) 

# x2=Lambda(my_reshape,arguments={'a':750*2,'b':4*3})(inpt)

    
def pk_model(time_input,num=1,nb_filter=8, kernel_size=(7,1),depths=5):
    
    inpt = Input(shape=time_input,name='input')
    # Down/Encode
    convs=[None]*depths
    net = Conv2d_BN1(inpt, nb_filter, kernel_size)
    for depth in range(depths):
        filters=int(2**depth*nb_filter)
        
        net = Conv2d_BN1(net, filters, kernel_size)
        convs[depth] = net
    
        if depth < depths - 1:
            net = Conv2d_BN2(net, filters, kernel_size)
    # Reshape        
    net_shape = net.get_shape().as_list()       
    net=Reshape((net_shape[1],net_shape[3]))(net)
    # LSTM
    net = keras.layers.LSTM(units=filters, return_sequences=True)(net)
    # Attention
    at_x,wt = SeqSelfAttention(return_attention=True, attention_width= 20,  
                            attention_activation='relu',name='Atten')(net)         
    net=Reshape((net_shape[1],1,net_shape[3]))(at_x)    
             
    # Up/Decode
    net1=net
    for depth in range(depths-2,-1,-1):
        filters = int(2**(depth) * nb_filter)  
        net1 = Conv2dT_BN3(net1, filters, kernel_size)
        # skip and concat
        net1 =Lambda(crop_and_cut)([convs[depth], net1])
                
    outenv = Conv2D(num, kernel_size=(3,1),padding='same',name='pk')(net1)
    
    model = Model(inpt, [outenv],name='pk_model')
    return model   

# In[] main
if __name__ == '__main__':
    args = read_args()
    start_gpu(args)
    if args.model_name=='bulid_pk_model':
        print(args.model_name)
        model=build_pk_model(time_input=(6000,3))
        model.summary()
        # data
        x_train=np.random.random((100,6000,3))
        y_train=np.max(abs(x_train),axis=2,keepdims=True)
        x_test=np.random.random((10,6000,3))
        y_test=np.max(abs(x_test),axis=2,keepdims=True)
              
    else:
        print(args.model_name)
        model=pk_model(time_input=(6000,1,3))
        model.summary() 
        # data
        x_train=np.random.random((100,6000,1,3))
        y_train=np.max(abs(x_train),axis=3,keepdims=True)
        x_test=np.random.random((10,6000,1,3))
        y_test=np.max(abs(x_test),axis=3,keepdims=True)  
        
        
    model.compile(loss=args.loss,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    saveBestModel= ModelCheckpoint('./model/%s.h5'%args.model_name, monitor=args.monitor, verbose=1, save_best_only=True,mode=args.monitor_mode)
    estop = EarlyStopping(monitor=args.monitor, patience=args.patience, verbose=0, mode=args.monitor_mode)
    callbacks_list = [saveBestModel,estop]
    # fit
    begin = datetime.datetime.now()
    history_callback=model.fit(x_train,y_train,
                               validation_data=(x_test,y_test),
                               batch_size=args.batch_size, epochs=args.epochs,verbose=1)

    #model.save_weights('./model/%s_wt.h5'%args.model_name) 
    end = datetime.datetime.now()
    print('Training time:',end-begin)    
    
    

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:05:54 2021

@author: alfah
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
#from keras import backend as K
from tensorflow.keras.layers import Conv2D#, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D,BatchNormalization, Bidirectional, LSTM, Dense, Lambda, Conv2DTranspose
from tensorflow.keras.layers import Average, Concatenate, ReLU, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional#, GRU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.compat.v1.keras.layers import CuDNNGRU as GRU

def transition(x, fils):
    x = Conv2D(fils, kernel_size = (1, 1), padding = 'same')(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = AveragePooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    
    return x
    
def dense_inception(x0, fils):
    x = inception(x0, fils)
    x = inception(Concatenate()([x, x0]), fils)
    
    return Concatenate()([x, x0])
    
def inception(x, fils):
    x1 = Conv2D(fils, kernel_size = (1, 1), padding = 'same')(x) #kernel_initializer = 'he_normal'
    #x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)
    x2 = Conv2D(fils, kernel_size = (1, 1), padding = 'same')(x)
    #x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)
    x3 = Conv2D(fils, kernel_size = (1, 1), padding = 'same')(x)
    #x3 = BatchNormalization()(x3)
    x3 = LeakyReLU()(x3)
    x4 = Conv2D(fils, kernel_size = (1, 1), padding = 'same')(x)
    #x4 = BatchNormalization()(x4)
    x4 = LeakyReLU()(x4)
    x5 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    x2 = Conv2D(fils, kernel_size = (3, 2), padding = 'same')(x2)
    #x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)
    x3 = Conv2D(fils, kernel_size = (5, 3), padding = 'same')(x3)
    #x3 = BatchNormalization()(x3)
    x3 = LeakyReLU()(x3)
    x4 = Conv2D(fils, kernel_size = (7, 5), padding = 'same')(x4)
    #x4 = BatchNormalization()(x4)
    x4 = LeakyReLU()(x4)
    x5 = Conv2D(fils, kernel_size = (1, 1), padding = 'same')(x5)
    #x5 = BatchNormalization()(x5)
    x5 = LeakyReLU()(x5)
    x2 = Conv2D(fils, kernel_size = (2, 3), padding = 'same')(x2)
    #x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)
    x3 = Conv2D(fils, kernel_size = (3, 5), padding = 'same')(x3)
    #x3 = BatchNormalization()(x3)
    x3 = LeakyReLU()(x3)
    x4 = Conv2D(fils, kernel_size = (7, 5), padding = 'same')(x4)
    #x4 = BatchNormalization()(x4)
    x4 = LeakyReLU()(x4)
    
    return Concatenate()([x1, x2, x3, x4, x5])

def getSinonet(train = True):
    inputShape = (725, 180, 1)
    filters = [32, 64, 128, 256, 512, 1024, 2048]
    inputs = Input(name = 'inputX', shape = inputShape, dtype = 'float32')
    x = Conv2D(filters[0], kernel_size = 3, kernel_initializer = 'he_normal', strides = (2, 2))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters[1], kernel_size = 3, kernel_initializer = 'he_normal', strides = (2, 2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2))(x)
    x = dense_inception(x, filters[2])
    x = transition(x, filters[3])
    x = dense_inception(x, filters[3])
    x = transition(x, filters[4])
    x = dense_inception(x, filters[4])
    x = GlobalAveragePooling2D(data_format = 'channels_last')(x)
    x = Dense(120, kernel_initializer = 'he_normal', name = 'features')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    output = Dense(6, kernel_initializer = 'he_normal', activation = 'softmax', name = 'output')(x)
    
    if train:
        return Model(inputs = inputs, outputs = output)
    else:
        return Model(inputs = [inputs], outputs = [output, x])
    
def getSinonet1(shape, train = True):
    inputShape = shape
    filters = [32, 64, 128, 256, 512, 1024, 2048]
    inputs = Input(name = 'inputX', shape = inputShape, dtype = 'float32')
    x = Conv2D(filters[0], kernel_size = 3, strides = (2, 2))(inputs)
    #x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters[1], kernel_size = 3, strides = (2, 2))(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2))(x)
    x = dense_inception(x, filters[2])
    x = transition(x, filters[3])
    x = dense_inception(x, filters[3])
    x = transition(x, filters[4])
    x = dense_inception(x, filters[4])
    x = GlobalAveragePooling2D(data_format = 'channels_last')(x)
    x = Dense(120, name = 'features')(x)
    #x = BatchNormalization()(x)
    x = ReLU()(x)
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    if train:
        return Model(inputs = inputs, outputs = output)
    else:
        return Model(inputs = [inputs], outputs = [output, x])
    
def getSinonet2(shape): #from_paper
    inputShape = shape
    filters = [32, 64, 128, 256, 512, 1024, 2048]
    inputs = Input(name = 'inputX', shape = inputShape, dtype = 'float32')
    x = Conv2D(filters[0], kernel_size = 3, strides = (2, 2))(inputs)
    #x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters[1], kernel_size = 3, strides = (2, 2))(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size = (3, 3), strides = (2, 2))(x)
    x = dense_inception(x, filters[2])
    x = transition(x, filters[3])
    x = dense_inception(x, filters[3])
    x = transition(x, filters[4])
    x = dense_inception(x, filters[4])
    x = GlobalAveragePooling2D(data_format = 'channels_last')(x)
    #x = Dense(120, name = 'features')(x)
    #x = BatchNormalization()(x)
    #x = ReLU()(x)
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = inputs, outputs = output)

def nonwin2win1(shape):
    inputShape = shape
    filters = [32, 64, 128, 256, 512, 1024, 2048]
    inputs = Input(name = 'inputX', shape = inputShape, dtype = 'float32')
    x = Conv2D(16, kernel_size = (3, 3), padding = 'same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(1, kernel_size = (3, 3), padding = 'same')(x)
    x = BatchNormalization()(x)
    outputs = ReLU()(x)
    
    return Model(inputs = inputs, outputs = outputs)

# define an encoder block
def define_encoder_block(layer_in, n_filters, filter_size = (3, 3), padding = 'valid', batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    f = Conv2D(n_filters, filter_size, strides=(1,1), padding= padding, kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        f = BatchNormalization()(f, training=True)
    # leaky relu activation
    f = LeakyReLU(alpha=0.2)(f)
    g = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(f)
    return f, g
 
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, filter_size = (3, 3), padding = 'valid', dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = UpSampling2D()(layer_in)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    g = Conv2DTranspose(n_filters, filter_size, strides=(1,1), padding= padding, kernel_initializer=init)(g)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # leakyrelu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g
 
# define the standalone generator model
def nonwin2win2(image_shape=(362,180,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape, dtype = 'float32')
	# encoder model
	c1, e1 = define_encoder_block(in_image, 64, padding='same', batchnorm=False)
	c2, e2 = define_encoder_block(e1, 128, (4,3))
	c3, e3 = define_encoder_block(e2, 256, (4,3))
	c4, e4 = define_encoder_block(e3, 512, (2,2))
	c5, e5 = define_encoder_block(e4, 512, (4,3))
	c6, e6 = define_encoder_block(e5, 512, (2,1))
	c7, e7 = define_encoder_block(e6, 512, padding='same')
	# decoder model
	d1 = decoder_block(e7, c7, 512, padding='same')
	d2 = decoder_block(d1, c6, 512, (2,1))
	d3 = decoder_block(d2, c5, 512, (4,3))
	d4 = decoder_block(d3, c4, 512, (2,2), dropout=False)
	d5 = decoder_block(d4, c3, 256, (4,3), dropout=False)
	d6 = decoder_block(d5, c2, 128, (4,3), dropout=False)
	d7 = decoder_block(d6, c1, 64, padding='same', dropout=False)
	# output
	g = Conv2D(1, (3,3), strides=(1,1), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('sigmoid', dtype = 'float32')(g)
	# define model
	model = Model(in_image, out_image)
	return model

"""  
sinonet_model = getSinonet()
#model.get_layer("features").output - for last feature maps
sinonet_model.summary()   
"""
def getLSTM(): #pre-concatenate - BiLSTM
    inputShapeA = (None, 120) #(batch_size, time_steps, 120)
    inputShapeB = (None, 1) #(batch_size, time_steps, 1)
    inputA = Input(name = 'inputA', shape = inputShapeA, dtype = 'float32')
    preds = Input(name = 'inputB', shape = inputShapeB, dtype = 'float32')
    x = Concatenate()([inputA, preds])
    x = Bidirectional(LSTM(256, return_sequences = True))(x) #merge_mode = 'ave', , dropout = 0.3
    x = Bidirectional(LSTM(256, return_sequences = True))(x)
    x = Bidirectional(LSTM(256, return_sequences = True))(x)
    x = Concatenate()([x, preds])
    #x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU()(x) #LeakyReLU
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = [inputA, preds], outputs = output)
"""
lstm_model = getLSTM()
lstm_model.summary()
"""
def getLSTM1(): #3BiLSTM
    inputShapeA = (None, 120) #(batch_size, time_steps, 120)
    inputShapeB = (None, 1) #(batch_size, time_steps, 1)
    inputA = Input(name = 'inputA', shape = inputShapeA, dtype = 'float32')
    preds = Input(name = 'inputB', shape = inputShapeB, dtype = 'float32')
    #x = Concatenate()([inputA, preds])
    x = Bidirectional(LSTM(256, return_sequences = True))(inputA) #merge_mode = 'ave', , dropout = 0.3
    x = Bidirectional(LSTM(256, return_sequences = True))(x)
    x = Bidirectional(LSTM(256, return_sequences = True))(x)
    x = Concatenate()([x, preds])
    #x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU()(x) #LeakyReLU
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = [inputA, preds], outputs = output)

def getLSTM2(): #3BiLSTM+attention_mechanism
    inputShapeA = (None, 120) #(batch_size, time_steps, 120)
    inputShapeB = (None, 1) #(batch_size, time_steps, 1)
    inputA = Input(name = 'inputA', shape = inputShapeA, dtype = 'float32')
    preds = Input(name = 'inputB', shape = inputShapeB, dtype = 'float32')
    #x = Concatenate()([inputA, preds])
    x = Bidirectional(LSTM(256, return_sequences = True))(inputA) #merge_mode = 'ave', , dropout = 0.3
    x = Bidirectional(LSTM(256, return_sequences = True))(x)
    x, h_f, _, h_b, _ = Bidirectional(LSTM(256, return_sequences = True, return_state = True))(x)
    h = Concatenate()([h_f, h_b])
    h = tf.expand_dims(h, axis = 1)
    score = Dense(1)(tf.nn.tanh(Dense(256)(x) + Dense(256)(h)))
    attention_weights = tf.nn.softmax(score, axis=1)
    x = attention_weights * x
    x = Concatenate()([x, preds])
    #x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU()(x) #LeakyReLU
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = [inputA, preds], outputs = output)

def getLSTM3(): #4BiLSTM+attention_mechanism
    inputShapeA = (None, 120) #(batch_size, time_steps, 120)
    inputShapeB = (None, 1) #(batch_size, time_steps, 1)
    inputA = Input(name = 'inputA', shape = inputShapeA, dtype = 'float32')
    preds = Input(name = 'inputB', shape = inputShapeB, dtype = 'float32')
    #x = Concatenate()([inputA, preds])
    x = Bidirectional(LSTM(256, return_sequences = True))(inputA) #merge_mode = 'ave', , dropout = 0.3
    x = Bidirectional(LSTM(256, return_sequences = True))(x)
    x = Bidirectional(LSTM(256, return_sequences = True))(x)
    x, h_f, _, h_b, _ = Bidirectional(LSTM(256, return_sequences = True, return_state = True))(x)
    h = Concatenate()([h_f, h_b])
    h = tf.expand_dims(h, axis = 1)
    score = Dense(1)(tf.nn.tanh(Dense(256)(x) + Dense(256)(h)))
    attention_weights = tf.nn.softmax(score, axis=1)
    x = attention_weights * x
    x = Concatenate()([x, preds])
    #x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU()(x) #LeakyReLU
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = [inputA, preds], outputs = output)

def getGRU(): #3BiGRU
    inputShapeA = (None, 120) #(batch_size, time_steps, 120)
    inputShapeB = (None, 1) #(batch_size, time_steps, 1)
    inputA = Input(name = 'inputA', shape = inputShapeA, dtype = 'float32')
    preds = Input(name = 'inputB', shape = inputShapeB, dtype = 'float32')
    #x = Concatenate()([inputA, preds])
    x = Bidirectional(GRU(256, return_sequences = True))(inputA) #merge_mode = 'ave', , dropout = 0.3
    x = Bidirectional(GRU(256, return_sequences = True))(x)
    x = Bidirectional(GRU(256, return_sequences = True))(x)
    x = Concatenate()([x, preds])
    #x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU()(x) #LeakyReLU
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = [inputA, preds], outputs = output)

def getGRU1(): #3BiGRU+attention_mechanism+afterGRU
    inputShapeA = (None, 120) #(batch_size, time_steps, 120)
    inputShapeB = (None, 1) #(batch_size, time_steps, 1)
    inputA = Input(name = 'inputA', shape = inputShapeA, dtype = 'float32')
    preds = Input(name = 'inputB', shape = inputShapeB, dtype = 'float32')
    #x = Concatenate()([inputA, preds])
    x = Bidirectional(GRU(256, return_sequences = True))(inputA) #merge_mode = 'ave', , dropout = 0.3
    x = Bidirectional(GRU(256, return_sequences = True))(x)
    x, h_f, h_b = Bidirectional(GRU(256, return_sequences = True, return_state = True))(x)
    h = Concatenate()([h_f, h_b])
    h = tf.expand_dims(h, axis = 1)
    score = Dense(1)(tf.nn.tanh(Dense(256)(x) + Dense(256)(h)))
    attention_weights = tf.nn.softmax(score, axis=1)
    x = attention_weights * x
    x = Concatenate()([x, preds])
    #x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU()(x) #LeakyReLU
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = [inputA, preds], outputs = output)

def getGRU2(): #3BiGRU+attention_mechanism+beforeLastGRU
    inputShapeA = (None, 120) #(batch_size, time_steps, 120)
    inputShapeB = (None, 1) #(batch_size, time_steps, 1)
    inputA = Input(name = 'inputA', shape = inputShapeA, dtype = 'float32')
    preds = Input(name = 'inputB', shape = inputShapeB, dtype = 'float32')
    x = Bidirectional(GRU(256, return_sequences = True))(inputA) #merge_mode = 'ave', , dropout = 0.3
    x, h_f, h_b = Bidirectional(GRU(256, return_sequences = True, return_state = True))(x)
    h = Concatenate()([h_f, h_b])
    h = tf.expand_dims(h, axis = 1)
    score = Dense(1)(tf.nn.tanh(Dense(256)(x) + Dense(256)(h)))
    attention_weights = tf.nn.softmax(score, axis=1)
    x = attention_weights * x
    x = Bidirectional(GRU(256, return_sequences = True))(x)
    x = Concatenate()([x, preds])
    #x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU()(x) #LeakyReLU
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = [inputA, preds], outputs = output)

def getGRU3(): #3BiGRU+attention_mechanism_after_every_layer
    inputShapeA = (None, 120) #(batch_size, time_steps, 120)
    inputShapeB = (None, 1) #(batch_size, time_steps, 1)
    inputA = Input(name = 'inputA', shape = inputShapeA, dtype = 'float32')
    preds = Input(name = 'inputB', shape = inputShapeB, dtype = 'float32')
    #x = Concatenate()([inputA, preds])
    x, h_f, h_b = Bidirectional(GRU(256, return_sequences = True, return_state = True))(inputA) #merge_mode = 'ave', , dropout = 0.3
    h = Concatenate()([h_f, h_b])
    h = tf.expand_dims(h, axis = 1)
    score = Dense(1)(tf.nn.tanh(Dense(256)(x) + Dense(256)(h)))
    attention_weights = tf.nn.softmax(score, axis=1)
    x = attention_weights * x
    x, h_f, h_b = Bidirectional(GRU(256, return_sequences = True, return_state = True))(x)
    h = Concatenate()([h_f, h_b])
    h = tf.expand_dims(h, axis = 1)
    score = Dense(1)(tf.nn.tanh(Dense(256)(x) + Dense(256)(h)))
    attention_weights = tf.nn.softmax(score, axis=1)
    x = attention_weights * x
    x, h_f, h_b = Bidirectional(GRU(256, return_sequences = True, return_state = True))(x)
    h = Concatenate()([h_f, h_b])
    h = tf.expand_dims(h, axis = 1)
    score = Dense(1)(tf.nn.tanh(Dense(256)(x) + Dense(256)(h)))
    attention_weights = tf.nn.softmax(score, axis=1)
    x = attention_weights * x
    x = Concatenate()([x, preds])
    #x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU()(x) #LeakyReLU
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = [inputA, preds], outputs = output)

def getGRU4(): #4BiGRU+attention_mechanism+afterGRU
    inputShapeA = (None, 120) #(batch_size, time_steps, 120)
    inputShapeB = (None, 1) #(batch_size, time_steps, 1)
    inputA = Input(name = 'inputA', shape = inputShapeA, dtype = 'float32')
    preds = Input(name = 'inputB', shape = inputShapeB, dtype = 'float32')
    #x = Concatenate()([inputA, preds])
    x = Bidirectional(GRU(256, return_sequences = True))(inputA) #merge_mode = 'ave', , dropout = 0.3
    x = Bidirectional(GRU(256, return_sequences = True))(x)
    x = Bidirectional(GRU(256, return_sequences = True))(x)
    x, h_f, h_b = Bidirectional(GRU(256, return_sequences = True, return_state = True))(x)
    h = Concatenate()([h_f, h_b])
    h = tf.expand_dims(h, axis = 1)
    score = Dense(1)(tf.nn.tanh(Dense(256)(x) + Dense(256)(h)))
    attention_weights = tf.nn.softmax(score, axis=1)
    x = attention_weights * x
    x = Concatenate()([x, preds])
    #x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = ReLU()(x) #LeakyReLU
    output = Dense(1, activation = 'sigmoid', name = 'output')(x)
    
    return Model(inputs = [inputA, preds], outputs = output)
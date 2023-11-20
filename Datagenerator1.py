#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:26:23 2021

@author: sindhura
"""

import numpy as np
import pandas as pd
import pydicom
import zipfile
import os
from skimage.transform import radon
import tensorflow as tf
import cv2

#su = 0
#sq = 0
class CNNDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, path, df, shape = (725, 360, 1), batch_size=32, num_classes=None, shuffle=True):
        self.path = path
        self.shape = shape
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        #self.x_col = x_col
        #self.y_col = y_col
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) // self.batch_size)

    def __getitem__(self, index):
        indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in indexes]
        #print(batch)
        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        #global su, sq
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
        #print('\n',su, sq)
        #su = 0
        #sq = 0

    def __get_data(self, batch):
        #global su, sq
        #su += np.sum(np.array(batch))
        #sq += np.sum(np.square(np.array(batch)))
        X = np.zeros((len(batch), *self.shape), dtype = 'float32')# logic
        y = np.zeros((len(batch), self.num_classes), dtype = 'float32') # logic
        
        for i, id in enumerate(batch):
            s = np.load(self.path+'/'+self.df.loc[id, 'Image']+'.npy')# logic
            mean = np.mean(s)
            std = np.std(s)
            if abs(mean)>0.01 and abs(std-1)>0.01:
                print('yes') if id==0 else print('')
                s = s-mean
                s = s/std if std>0 else s
            if s.shape == self.shape:
                X[i,:,:,:] = s
            else:
                X[i,:,:,0] = s
            #y[i] = self.df.loc[id, 'any'].to_numpy(dtype = 'float32')
            y[i, ] = self.df.loc[id, ['any']].to_numpy(dtype = 'float32') # labels

        return X, y
    
class nonwin2winGenerator(tf.keras.utils.Sequence):
    def __init__(self,inputPath, outputPath, df, shape = (725, 360, 1), batch_size = 32, shuffle = True):
        self.path1 = inputPath
        self.path2 = outputPath
        self.df = df
        self.indices = self.df.index.tolist()
        self.shape = shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return (len(self.indices) // self.batch_size)

    def __getitem__(self, index):
        if index == len(self.indices) // self.batch_size - 1:
            indexes = self.index[index * self.batch_size: len(self.indices)]
        else:
            indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in indexes]
        
        X, Y = self.__get_data(batch)
        return X, Y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
            

    def __get_data(self, batch):
        X = np.zeros((len(batch), int(self.shape[0]/2), int(self.shape[1]/2), self.shape[2]), dtype = 'float32')# logic
        Y = np.zeros((len(batch), int(self.shape[0]/2), int(self.shape[1]/2), self.shape[2]), dtype = 'float32') # logic
        #z = []
        
        for i, id in enumerate(batch):
            x = np.load(self.path1+'/'+self.df.loc[id, 'Image']+'.npy')# inputs
            x1 = cv2.resize(x[:,:,0], (int(self.shape[1]/2), int(self.shape[0]/2)))
            X[i,:,:,0] = cv2.normalize(src=x1, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #X[i,] = (x - np.min(x))/(np.max(x) - np.min(x))
            y = np.load(self.path2+'/'+self.df.loc[id, 'Image']+'.npy')#labels
            y1 = cv2.resize(y[:,:,0], (int(self.shape[1]/2), int(self.shape[0]/2)))
            Y[i,:,:,0] = cv2.normalize(src=y1, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #Y[i,] = (y - np.min(y))/(np.max(y) - np.min(y))

        return X, Y
    
class nonwin2winDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,inputPath, outputPath, df, shape = (725, 360, 1), batch_size = 32, shuffle = True):
        self.path1 = inputPath
        self.path2 = outputPath
        self.df = df
        self.indices = self.df.index.tolist()
        self.shape = shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return (len(self.indices) // self.batch_size)

    def __getitem__(self, index):
        if index == len(self.indices) // self.batch_size - 1:
            indexes = self.index[index * self.batch_size: len(self.indices)]
        else:
            indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in indexes]
        
        X, Y, Z = self.__get_data(batch)
        return X, Y, Z

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
            

    def __get_data(self, batch):
        X = np.zeros((len(batch), int(self.shape[0]/2), int(self.shape[1]/2), self.shape[2]), dtype = 'float32')# logic
        Y = np.zeros((len(batch), int(self.shape[0]/2), int(self.shape[1]/2), self.shape[2]), dtype = 'float32') # logic
        Z = []
        
        for i, id in enumerate(batch):
            x = np.load(self.path1+'/'+self.df.loc[id, 'Image']+'.npy')# inputs
            x1 = cv2.resize(x, (int(self.shape[1]/2), int(self.shape[0]/2)))
            meanx = np.mean(x1)
            stdx = np.std(x1)
            x1 = x1-meanx
            x1 = x1/stdx if stdx>0 else x1
            X[i,:,:,0] = cv2.normalize(src=x1, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            y = np.load(self.path2+'/'+self.df.loc[id, 'Image']+'.npy')#labels
            y1 = cv2.resize(y, (int(self.shape[1]/2), int(self.shape[0]/2)))
            meany = np.mean(y1)
            stdy = np.std(y1)
            y1 = y1-meany
            y1 = y1/stdy if stdy>0 else y1
            Y[i,:,:,0] = cv2.normalize(src=y1, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            Z.append(str(self.df.loc[id, 'Image']))

        return X, Y, Z

class CNNDataGeneratorEmbeds(tf.keras.utils.Sequence):
    def __init__(self, path, df, shape = (725, 360, 1), batch_size=32, num_classes=None, shuffle=True):
        self.path = path
        self.shape = shape
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        #self.x_col = x_col
        #self.y_col = y_col
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) // self.batch_size)

    def __getitem__(self, index):
        indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in indexes]
        
        X, y, z = self.__get_data(batch)
        return X, y, z

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X = np.zeros((len(batch), *self.shape), dtype = 'float32')# logic
        y = np.zeros((len(batch), self.num_classes), dtype = 'float32') # logic
        z = []
        
        for i, id in enumerate(batch):
            s = np.load(self.path+'/'+self.df.loc[id, 'Image']+'.npy')# logic
            mean = np.mean(s)
            std = np.std(s)
            if abs(mean)>0.01 and abs(std-1)>0.01:
                print('yes') if id==0 else print('')
                s = s-mean
                s = s/std if std>0 else s
            if s.shape == self.shape:
                X[i,:,:,:] = s
            else:
                X[i,:,:,0] = s
            #y[i] = self.df.loc[id, 'any'].to_numpy(dtype = 'float32')
            y[i, ] = self.df.loc[id, ['any']].to_numpy(dtype = 'float32') # labels
            z.append(str(self.df.loc[id, 'Image']))

        return X, y, z
    
class LSTMDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=1, num_classes=1, features = 120, shuffle=True):
        self.batch_size = batch_size
        self.df = df
        self.features = features
        self.SeriesInstanceUID = self.df['SeriesInstanceUID'].unique()
        self.indices = self.SeriesInstanceUID.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.embeds = [str(i) for i in range(self.features)]
        #self.x_col = x_col
        #self.y_col = y_col
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) // self.batch_size)

    def __getitem__(self, index):
        indexes = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in indexes]
        x, y, z = self.__get_data(batch)
        
        return [x, y], z

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        series_set = self.df[self.df['SeriesInstanceUID'].isin(batch)].sort_values(by=['ImagePositionSpan', 'ImageId'])
        series_set.reset_index(inplace = True, drop = True)
        
        x = np.zeros((self.batch_size, len(series_set), self.features), dtype = 'float32')# logic
        y = np.zeros((self.batch_size, len(series_set), self.num_classes), dtype = 'float32') # logic
        z = np.zeros((self.batch_size, len(series_set), self.num_classes), dtype = 'float32')
        
        x[0,:,:] = series_set[self.embeds].to_numpy()
        y[0,:,0] = series_set['0_x'].to_numpy()
        z[0,:,0] = series_set['any'].to_numpy()
        
        return x, y, z
    

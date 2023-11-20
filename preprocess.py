#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 19:35:34 2021

@author: sindhura
"""
#%%
#import libraries and create base csv df
import glob
import os
import numpy as np
import pandas as pd
import zipfile
from skimage.transform import radon
import pydicom
from tqdm import tqdm
import cv2

target = 'rsna-intracranial-hemorrhage-detection.zip'
handle = zipfile.ZipFile(target)

def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    #subdural_img = window_image(dcm, 80, 200)
    #soft_img = window_image(dcm, 40, 380)

    brain_img = (brain_img - 0) / 80
    #subdural_img = (subdural_img - (-20)) / 200
    #soft_img = (soft_img - (-150)) / 380
    bsb_img = np.squeeze(np.array([brain_img]).transpose(1, 2, 0))#, subdural_img, soft_img]).transpose(1, 2, 0)
    bsb_img = cv2.resize(bsb_img, (512, 512))
    theta = np.linspace(0., 180., 360)
    sinogram = np.array(radon(bsb_img, theta = theta, circle=False), dtype = 'float32')
    s = sinogram - np.mean(sinogram)
    s = s/np.std(sinogram) if np.std(sinogram)>0 else s
    #sinogram = sinogram/np.max(sinogram)

    return s

dir_csv = 'rsna-intracranial-hemorrhage-detection'
test_images_dir = 'stage_2_test'
train_images_dir = 'stage_2_train'
train_metadata_csv = 'train_metadata_noidx.csv'
test_metadata_csv = 'test_metadata_noidx.csv'

train_metadata_noidx = pd.read_csv(train_metadata_csv)

with handle.open('stage_2_train.csv') as t:
    train = pd.read_csv(t)

# Prepare train table
train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']

merged_train = pd.merge(left=train, right=train_metadata_noidx, how='left', left_on='Image', right_on='ImageId')
#%%
#classify CT scans into lists of ich-y and ich-n
df = merged_train.copy()
ich_y = []
ich_n = []
df.sort_values(['SeriesInstanceUID'])
train_series = train_metadata_noidx['SeriesInstanceUID'].unique()
for i in tqdm(range(len(train_series))):
    d = df[df['SeriesInstanceUID'] == train_series[i]]
    if np.sum(d['any']) > 0:
        ich_y.append(train_series[i])
    else:
        ich_n.append(train_series[i])
#%%
#create csv files for train, valid and test
train = ich_y[3000:3800]
valid = ich_y[3800:3900]
test = ich_y[3900:4000]
train_df = merged_train[merged_train['SeriesInstanceUID'].isin(train)]
valid_df = merged_train[merged_train['SeriesInstanceUID'].isin(valid)]
test_df = merged_train[merged_train['SeriesInstanceUID'].isin(test)]
train_df.to_csv('train4_800.csv', index=False)
print(train_df['any'].value_counts())
valid_df.to_csv('valid4_100.csv', index=False)
print(valid_df['any'].value_counts())
test_df.to_csv('test4_100.csv')
print(test_df['any'].value_counts())
#%%
#generate sinograms and save as .npy files
for i in tqdm(range(len(train))):
    d = df[df['SeriesInstanceUID'] == train[i]]
    d.reset_index(inplace = True, drop = True)
    for j in range(d.shape[0]):
        image = np.zeros((725, 360, 1), dtype = 'float32')
        with handle.open(os.path.join(train_images_dir, d.loc[j, 'Image'] + '.dcm')) as p:
            dicom = pydicom.dcmread(p)
            bb = bsb_window(dicom)
            bb = np.reshape(bb, (725, 360, 1))
            #if bb.shape == (725, 180, 1):
            image[:,:,:] = bb
            np.save(str(d.loc[j, 'Image'])+'.npy', image)
            #else:
                #t.append(d.loc[j, 'Image'])

print('train is over')
for ii in tqdm(range(len(valid))):
    d = df[df['SeriesInstanceUID'] == valid[ii]]
    d.reset_index(inplace = True, drop = True)
    for jj in range(d.shape[0]):
        image = np.zeros((725, 360, 1), dtype = 'float32')
        with handle.open(os.path.join(train_images_dir, d.loc[jj, 'Image'] + '.dcm')) as pp:
            dicom = pydicom.dcmread(pp)
            bb= bsb_window(dicom)
            #if bb.shape == (725, 180, 1):
            bb = np.reshape(bb, (725, 360, 1)) 
            image[:,:,:]  = bb
            np.save(str(d.loc[jj, 'Image'])+'.npy', image)
            #else:
              #  v.append(d.loc[jj, 'Image']

for j in tqdm(range(len(test))):
    d = df[df['SeriesInstanceUID'] == test[j]]
    d.reset_index(inplace = True, drop = True)
    for jj in range(d.shape[0]):
        image = np.zeros((725, 360, 1), dtype = 'float32')
        with handle.open(os.path.join(train_images_dir, d.loc[jj, 'Image'] + '.dcm')) as pp:
            dicom = pydicom.dcmread(pp)
            bb= bsb_window(dicom)
            #if bb.shape == (725, 180, 1):
            bb = np.reshape(bb, (725, 360, 1)) 
            image[:,:,:]  = bb
            np.save(str(d.loc[jj, 'Image'])+'.npy', image)

np.save('Alldone.npy',np.zeros((1, 1, 1)))
            
#%%
for i in range(len(train)):
    if i>=20:
        break
    else:
        d = df[df['SeriesInstanceUID'] == train[i]]
        d.reset_index(inplace = True, drop = True)
        print(d['any'].value_counts())

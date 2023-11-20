#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:41:28 2021

@author: sindhura
"""
#%%
import glob
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
#tf.config.run_functions_eagerly(True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import cv2
from CLR_master.clr_callback import *
#from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve
from tensorflow.keras.models import Model
import tqdm

from keras_models import getSinonet2, getSinonet1, nonwin2win1, nonwin2win2
from Datagenerator1 import CNNDataGenerator, CNNDataGeneratorEmbeds, nonwin2winGenerator, nonwin2winDataGenerator #IntracranialDataset1
from losses import MS_SSIM, MAE, PSNR

mixed_precision.set_global_policy('mixed_float16')
#%%
train = pd.read_csv('train8882.csv')
valid = pd.read_csv('valid8882.csv')
test = pd.read_csv('test8882.csv')
"""
test1 = pd.read_csv('CT-8882/test1_100.csv')
test2 = pd.read_csv('CT-8882/test2_100.csv')
test_n = pd.read_csv('CT-8882/test_ich-n_200.csv')

test = pd.concat([test1, test2, test_n], 
                  axis=0, ignore_index=True)
"""

#Data splitting and management
saved_model_dir = 'cnn_models3'#nonwin2win_cnn_models1, cnn_models3
try:
    os.mkdir(saved_model_dir)
except:
    pass

#inputPath = 'NWCT-8882'
#outputPath = 'CT-8882'
#path = 'CT-8882'
path = '../Resnext_sinogram/CT-8882'
#path = '../preprocess_files/Test8882_Offset_Error_W_Sinograms'
#inputPath = '../preprocess_files/Test8882_NW_Quant_Error_Sinograms'
#path = 'unet_preds_sinograms'
shape = (725, 360, 1)
#shape1 = (362, 180, 1)

n_classes = 1
n_epochs = 30
batch_size = 32
#batch_size = 64

print(train['any'].value_counts())
print(valid['any'].value_counts())
print(test['any'].value_counts())
"""
train_dataset = nonwin2winGenerator(df = train,
                                 inputPath = inputPath,
                                 outputPath = outputPath,
                                 shape = shape,
                                 batch_size = batch_size,
                                 shuffle = True)

valid_dataset = nonwin2winGenerator(df = valid,
                                 inputPath = inputPath,
                                 outputPath = outputPath,
                                 shape = shape,
                                 batch_size = batch_size,
                                 shuffle = False)

test_dataset = nonwin2winGenerator(df = test,
                                inputPath = inputPath,
                                outputPath = outputPath,
                                shape = shape,
                                batch_size = batch_size,
                                shuffle = False)

train_dataset = CNNDataGenerator(df = train,
                                 path = path,
                                 shape = shape,
                                 batch_size = batch_size,
                                 num_classes = n_classes,
                                 shuffle = True)
valid_dataset = CNNDataGenerator(df = valid,
                                 path = path,
                                 shape = shape,
                                 batch_size = batch_size,
                                 num_classes = n_classes,
                                 shuffle = False)
"""
test_dataset = CNNDataGenerator(df = test,
                                path = path,
                                shape = shape,
                                batch_size = 1,
                                num_classes = n_classes,
                                shuffle = False)

#print('len of train_dataset',len(train_dataset))
#print('len of valid_dataset',len(valid_dataset))
print('len of test_dataset',len(test_dataset))
#%%
model = None
del model
model = getSinonet2(shape=shape)
model.load_weights(saved_model_dir+'/Sinonet8882-1-028-0.886016.h5')# Sinonet8882-1-028-0.886016.h5, CNN8882-2-012-0.893494.h5, nonwin2win2-009-0.444055.h5
weights = np.array(model.get_weights())#for model weights pertubation
model.set_weights(weights + 0.001*weights)#for model weights pertubation
adadelta = tf.keras.optimizers.Adadelta(
    learning_rate=1.0, rho=0.95, epsilon=1e-07, name='Adadelta',
)

threshold = 0.5
def sensitivity_threshold(threshold=0.5):
    def sensitivity(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        def f1(): return tf.divide(true_positives, possible_positives)
        def f2(): return tf.constant(1.0, dtype=tf.float32)
        return tf.cond(tf.greater(possible_positives, 0), f1, f2)
    return sensitivity

def specificity_threshold(threshold=0.5):
    def specificity(y_true, y_pred):
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
        possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
        def f1(): return tf.divide(true_negatives, possible_negatives)
        def f2(): return tf.constant(1.0, dtype=tf.float32)
        return tf.cond(tf.greater(possible_negatives, 0), f1, f2)
    return specificity

auc = tf.keras.metrics.AUC()
"""
clr_triangular = CyclicLR(mode='triangular',
                          base_lr = 0.001,
                          max_lr = 1.0,
                          step_size = 23130)
"""
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("Learning Rate = ", K.eval(lr_with_decay))
"""
def Jaccard_img(y_true, y_pred): #https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    intersection = y_true*y_pred
    union = 1 -((1-y_true)*(1-y_pred))
    return (K.sum(intersection) / K.sum(union))

def dice_img(y_true, y_pred):
    intersection = y_true*y_pred
    return 2. * K.sum(intersection)/(K.sum(y_true) + K.sum(y_pred))

def jaccard_loss(y_true, y_pred): #https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    intersection = y_true*y_pred
    union = 1 -((1-y_true)*(1-y_pred))
    return 1-(K.sum(intersection) / K.sum(union))

def SSIM(y_true, y_pred):
    return tf.math.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def W_SUM_LOSS(y_true, y_pred):
    return 0.15*MAE(y_true, y_pred) + 0.85*tf.expand_dims(tf.expand_dims(MS_SSIM(y_true, y_pred), axis=[1]), axis=[1])

def Cos_Sim(y_true, y_pred):
    num = y_true * y_pred
    denom = tf.norm(y_true, axis = [-3, -2], keepdims = True) * tf.norm(y_pred, axis = [-3, -2], keepdims = True) + K.epsilon()
    cos = num/denom
    d = tf.squeeze(K.sum(cos, axis = [-3, -2], keepdims = True))
    return tf.math.reduce_mean(d)

def psnr(y_true, y_pred):
    return tf.math.reduce_mean(PSNR(y_true, y_pred))
#Jaccard_img, dice_img, SSIM, Cos_Sim, psnr, loss = W_SUM_LOSS
"""
model.compile(optimizer = adadelta, loss = 'binary_crossentropy', metrics = ['accuracy', auc, sensitivity_threshold(threshold), specificity_threshold(threshold)]) #, auc, sensitivity_threshold(threshold), specificity_threshold(threshold
#%%
checkpoint = ModelCheckpoint(saved_model_dir+'/nonwin2win2_2-{epoch:03d}-{val_SSIM:03f}.h5', monitor = 'val_SSIM', verbose = 0, save_best_only = True,
    save_weights_only = True, mode = 'max')

#reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
#                              patience=1, mode = 'max')

csv_logger = CSVLogger(saved_model_dir+'/nonwin2win2_2_training_history.csv', append=True, separator=',')

model.fit(train_dataset,
            epochs = n_epochs,
            validation_data = valid_dataset,
            callbacks = [checkpoint,  MyCallback(), csv_logger],#MyCallback(),
            use_multiprocessing = False,
            workers = 1,
            verbose = 1,
            initial_epoch = 0)
#%%
#model.save_weights('cnn_models/CNN4-50weights.h5')
model.evaluate(test_dataset, 
               batch_size = batch_size)
#%%
test_dataset1 = nonwin2winDataGenerator(df = test,
                                inputPath = inputPath,
                                outputPath = outputPath,
                                shape = shape,
                                batch_size = 1,
                                shuffle = False)
unet_preds_dir = 'unet_test8882_quant_error_sinograms' 
try:
    os.mkdir(unet_preds_dir)
except:
    pass

for i, (x, y, z) in enumerate(test_dataset1):
    pred = model.predict([x])[0,:,:,0]
    pred = cv2.resize(pred, (int(shape[1]), int(shape[0])))
    pred = cv2.normalize(src=pred, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    pred = np.reshape(pred, shape)
    s = pred - np.mean(pred)
    s = s/np.std(pred) if np.std(pred)>0 else s
    np.save(unet_preds_dir+'/'+str(z[0])+'.npy', s)

#%%
batch_size = 1
y_preds = np.zeros((len(test), 1), dtype = 'float32')
y_trues = np.zeros((len(test), 1), dtype = 'float32')
Y_preds = np.zeros((len(test), 1), dtype = 'float32')
Y_trues = np.zeros((len(test), 1), dtype = 'float32')
l=0
fn = []#false negatives
fp = []#false positives
cp = []
for i, (x, z) in enumerate(test_dataset):
    y_true = z
    batch_size = y_true.shape[0]
    y_pred = model.predict([x])
    #y_pred = 1.0 if y_pred>0.5 else 0.0
    """
    if y_pred == 1.0 and y_true == 0.0:
        fp.append(i)
    if y_pred == 0.0  and y_true == 1.0:
        fn.append(i)
    if y_pred == y_true:
        cp.append(i)
    """
    y_preds[l:l+batch_size,0] = np.squeeze(y_pred)
    y_trues[l:l+batch_size,0] = np.squeeze(y_true)
    l += batch_size
"""
f_p = np.array(fp)
f_n = np.array(fn)
np.save('f_n_sinonet.npy', f_n)
np.save('f_p_sinonet.npy', f_p)
c_p = np.array(cp)
np.save('c_p_mod-sinonet.npy', c_p)
"""

thresh = 0.5
for i in range(len(y_preds)):
    Y_preds[i] = 0.0 if y_preds[i]<=thresh else 1.0

conf_matrix = tf.math.confusion_matrix(np.squeeze(y_trues), np.squeeze(Y_preds), 2).numpy()
print('Threshold: ', thresh)
print('Confusion matrix: ', conf_matrix)
print('Sensitivity: ', (conf_matrix[1, 1]/(conf_matrix[1, 1]+conf_matrix[1, 0]))*100)
print('Specificity: ', (conf_matrix[0, 0]/(conf_matrix[0, 0]+conf_matrix[0, 1]))*100)
print('Accuracy: ', (conf_matrix[0, 0]+conf_matrix[1, 1])/(np.sum(conf_matrix))*100)
auc.reset_states()
auc.update_state(y_trues, y_preds)
print('AUC: ',auc.result().numpy())
#%%
inputs, labels = test_dataset.__getitem__(1)
preds = model.predict(inputs)
#%%
fpr, tpr, thresholds = roc_curve(y_trues, y_preds, pos_label=1)
np.save('roc_curve/fpr_CNN8882.npy', fpr)
np.save('roc_curve/tpr_CNN8882.npy', tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
#%%
#storing preds and last layer's features
modelx = None
layer = 'features'
modelx = getSinonet1(shape, False)
modelx.load_weights(saved_model_dir+'/CNN8882-2-012-0.893494.h5')
weights = np.array(modelx.get_weights())#for model weights pertubation
modelx.set_weights(weights + 0.001*weights)#for model weights pertubation
inter_layer_model = Model(inputs=modelx.input,
                                 outputs=modelx.get_layer(layer).output)

#tf.config.run_functions_eagerly(True)

"""
train_embeds = CNNDataGeneratorEmbeds(df = train,
                                 path = path,
                                 shape = shape,
                                 batch_size = batch_size,
                                 num_classes = n_classes,
                                 shuffle = False)
valid_embeds = CNNDataGeneratorEmbeds(df = valid,
                                 path = path,
                                 shape = shape,
                                 batch_size = batch_size,
                                 num_classes = n_classes,
                                 shuffle = False)
"""
test_embeds = CNNDataGeneratorEmbeds(df = test,
                                path = path,
                                shape = shape,
                                batch_size = batch_size,
                                num_classes = n_classes,
                                shuffle = False)

#print('len of train_dataset',len(train_embeds))
#print('len of valid_dataset',len(valid_embeds))
print('len of test_dataset',len(test_embeds))
#%%
new_dir = 'test8882_weights_quant_error_features_preds'
try:
    os.mkdir(new_dir)
except:
    pass
#%%

train_pred_dict = {}
train_embed_dict = {}

for i, (x, y, z) in enumerate(train_embeds):
    #preds, _ = modelx.predict(x)
    embeds = inter_layer_model.predict(x)
    #preds = sigmoid(preds).numpy()
    embeds = np.squeeze(embeds)
    for a,  c in zip(z,  embeds): #preds,b,
        #train_pred_dict[a] = b
        train_embed_dict[a] = c
    
train_embed_df = pd.DataFrame.from_dict(train_embed_dict, orient='index')
train_embed_df.to_csv(new_dir+'/Sinonet1_train_embeds.csv')
"""
train_pred_df = pd.DataFrame.from_dict(train_pred_dict, orient='index')
train_pred_df.to_csv(new_dir+'/Sinonet1_train_preds.csv')
"""
valid_pred_dict = {}
valid_embed_dict = {}

for i, (x, y, z) in enumerate(valid_embeds):
    #preds, _ = modelx.predict(x)
    embeds = inter_layer_model.predict(x)
    #preds = sigmoid(preds).numpy()
    embeds = np.squeeze(embeds)
    for a,  c in zip(z,  embeds): #preds,b,
       #valid_pred_dict[a] = b
        valid_embed_dict[a] = c
    
valid_embed_df = pd.DataFrame.from_dict(valid_embed_dict, orient='index')
valid_embed_df.to_csv(new_dir+'/Sinonet1_valid_embeds.csv')
"""
valid_pred_df = pd.DataFrame.from_dict(valid_pred_dict, orient='index')
valid_pred_df.to_csv(new_dir+'/Sinonet1_valid_preds.csv')
"""
#%%
#test_pred_dict = {}
test_embed_dict = {}

for i, (x, y, z) in enumerate(test_embeds):
    #preds, embeds = modelx.predict(x)
    embeds = inter_layer_model.predict(x)
    #preds = sigmoid(preds).numpy()
    #embeds = np.squeeze(embeds)
    for a,  c in zip(z, embeds): #preds, b,
        #test_pred_dict[a] = b
        test_embed_dict[a] = c
    
test_embed_df = pd.DataFrame.from_dict(test_embed_dict, orient='index')
test_embed_df.to_csv(new_dir+'/test8882_weights_quant_error_features.csv')
"""
test_pred_df = pd.DataFrame.from_dict(test_pred_dict, orient='index')
test_pred_df.to_csv(new_dir+'/test8882_weights_quant_error_preds.csv')
"""
#%%
for i, (x, y) in enumerate(train_dataset):
    if i>=1:
        break
    else:
        print(x.shape)
        print(y.shape)
#%%
import numpy as np
c_p = list(np.load('c_p_mod-sinonet.npy'))
f_p_s = list(np.load('f_p_sinonet.npy'))
f_n_s = list(np.load('f_n_sinonet.npy'))
f_p_r = list(np.load('f_p_resnext.npy'))
f_n_r = list(np.load('f_n_resnext.npy'))
f_p = list(set(f_p_s) & set(f_p_r))
f_n = list(set(f_n_s) & set(f_n_r))
final_fp = list(set(c_p) & set(f_p))
final_fn = list(set(c_p) & set(f_n))
final_fp.sort()
final_fn.sort()
#%%
import pydicom
import zipfile
import cv2

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
    
    return bsb_img

path = 'CT-8882'
train_images_dir = 'stage_2_train'
target = 'rsna-intracranial-hemorrhage-detection.zip'
test = pd.read_csv('test8882.csv')
handle = zipfile.ZipFile(target)
#%%
#61, 73, 76, 91
idx = 91
with handle.open(os.path.join(train_images_dir, str(test.loc[final_fn[idx], 'Image'])+'.dcm')) as p:
    dicom = pydicom.dcmread(p)
    img = bsb_window(dicom)

image = np.load(path+'/'+test.loc[final_fn[idx], 'Image']+'.npy')
image = np.squeeze(image)
u8_img4 = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
u8_image4 = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#%%
fig, axs = plt.subplots(2, 4, figsize=(18, 8))

#ax1.set_title("Original")
for i in range(4):
    axs[0,i].set_xticks([])
    axs[0,i].set_yticks([])
axs[0,0].imshow(u8_img1, cmap=plt.cm.Greys_r, aspect = 'auto')
axs[0,1].imshow(u8_img2, cmap=plt.cm.Greys_r, aspect = 'auto')
axs[0,2].imshow(u8_img3, cmap=plt.cm.Greys_r, aspect = 'auto')
am1 = axs[0,3].imshow(u8_img4, cmap=plt.cm.Greys_r, aspect = 'auto')

dx, dy = 0.5 * 180.0 / max(img.shape), 0.5 / image.shape[0]
#ax2.set_title("Radon transform\n(Sinogram)")
axs[1,0].set_xlabel("Projection angle (deg)", fontsize = 13)
axs[1,0].set_ylabel("Projection position (pixels)", fontsize = 13)
axs[1,0].imshow(u8_image1, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, image.shape[0] + dy),
           aspect='auto')
axs[1,1].imshow(u8_image2, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, image.shape[0] + dy),
           aspect='auto')
axs[1,2].imshow(u8_image3, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, image.shape[0] + dy),
           aspect='auto')
am2 = axs[1,3].imshow(u8_image4, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, image.shape[0] + dy),
           aspect='auto')

fig.tight_layout(pad=5)
fig.colorbar(am1, ax = axs[0,3], shrink=1)
fig.colorbar(am2, ax = axs[1,3])
#plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
#plt.xlabel('Projection angle (deg)')
#fig.text(0.5, 0.04, 'Projection angle (deg)', va='center', ha='center')#, fontsize=rcParams['axes.labelsize']
plt.show()
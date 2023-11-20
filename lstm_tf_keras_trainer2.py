#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 03:59:43 2021

@author: sindhura
"""
#%%
import os
import numpy as np
import pandas as pd
#from sklearn.metrics import roc_auc_score, classification_report
#import zipfile
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, Callback, LearningRateScheduler
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from Datagenerator1 import LSTMDataGenerator
from keras_models import getGRU4, getGRU1, getLSTM2, getLSTM3, getGRU, getLSTM1

mixed_precision.set_global_policy('mixed_float16')
#%%
#COLS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
#saved_model_dir = 'new_lstm_models2_wo_schdlr1'
saved_model_dir = 'lstm_models2_CuDnn2'
try:
    os.mkdir(saved_model_dir)
except:
    pass

dir_csv = 'rsna-intracranial-hemorrhage-detection'
test_images_dir = 'stage_2_test'
train_images_dir = 'stage_2_train'
train_metadata_csv = 'train_metadata_noidx.csv'
test_metadata_csv = 'test_metadata_noidx.csv'
train_path = 'train8882.csv'#CT3-1000/train3_800.csv
valid_path = 'valid8882.csv'
test_path = 'test8882.csv'
#path = 'nonwin2win_features_preds'
path = 'test8882_weights_quant_error_features_preds'
#path = 'CNN8882_preds_embeds'
#path = 'Sinonet1_features_preds'
#path = 'test'
#shape = (725, 360, 1)

n_classes = 1
n_epochs = 50
batch_size = 1
features = 120


test_preds = pd.read_csv(path+'/test8882_weights_quant_error_preds.csv', index_col=False)
#valid_preds = pd.read_csv(path+'/CNN8882_valid_preds.csv', index_col=False)
#train_preds = pd.read_csv(path+'/CNN8882_train_preds.csv', index_col=False)

test_preds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)
#valid_preds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)
#train_preds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)

#valid_embeds = pd.read_csv(path+'/Sinonet1_valid_embeds.csv', index_col=False)
#train_embeds = pd.read_csv(path+'/Sinonet1_train_embeds.csv', index_col=False)
test_embeds = pd.read_csv(path+'/test8882_weights_quant_error_features.csv', index_col=False)

#valid_embeds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)
#train_embeds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)
test_embeds.rename(columns={'Unnamed: 0': 'Image'}, inplace=True)

train = pd.read_csv(train_path)
valid = pd.read_csv(valid_path)
test = pd.read_csv(test_path)

del test['Unnamed: 0.1']

#merged_train = pd.merge(left=train_preds, right=train, how='left', left_on='Image', right_on='Image')
#merged_valid = pd.merge(left=valid_preds, right=valid, how='left', left_on='Image', right_on='Image')
merged_test = pd.merge(left=test_preds, right=test, how='left', left_on='Image', right_on='Image')


#merged_train = pd.merge(left=merged_train, right=train_embeds, how='left', left_on='Image', right_on='Image')
#merged_valid = pd.merge(left=merged_valid, right=valid_embeds, how='left', left_on='Image', right_on='Image')
merged_test = pd.merge(left=merged_test, right=test_embeds, how='left', left_on='Image', right_on='Image')

#merged_train.rename(columns={'0_y':'0'}, inplace=True)
#merged_valid.rename(columns={'0_y':'0'}, inplace=True)
merged_test.rename(columns={'0_y':'0'}, inplace=True)

#print(merged_train['any'].value_counts())
#print(merged_valid['any'].value_counts())
print(merged_test['any'].value_counts())
#%%
"""
train_dataset = LSTMDataGenerator(merged_train,
                                  batch_size = batch_size,
                                  num_classes = n_classes,
                                  features = features,
                                  shuffle = True)
valid_dataset = LSTMDataGenerator(merged_valid,
                                  batch_size = batch_size,
                                  num_classes = n_classes,
                                  features = features,
                                  shuffle = False)
"""
test_dataset = LSTMDataGenerator(merged_test,
                                  batch_size = batch_size,
                                  num_classes = n_classes,
                                  features = features,
                                  shuffle = False)

#print('len of train_dataset',len(train_dataset))
#print('len of valid_dataset',len(valid_dataset))
print('len of test_dataset',len(test_dataset))
#%%
model = None
del model 
model = getLSTM2()
model.load_weights(saved_model_dir+'/LSTM2-024-0.906239.h5')
weights = np.array(model.get_weights())#for model weights pertubation
model.set_weights(weights + 0.001*weights)#for model weights pertubation
auc = tf.keras.metrics.AUC()
auc.reset_states()
"""
exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    1.0,
    decay_steps=25000,
    decay_rate=0.96,
    staircase=True)
"""
adadelta = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-07, name='Adadelta', decay = 1e-07)

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

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("Learning Rate = ", K.eval(lr_with_decay))

model.compile(optimizer = adadelta, loss = 'binary_crossentropy', 
              metrics = ['accuracy', auc, sensitivity_threshold(threshold), specificity_threshold(threshold)])
#%%
checkpoint = ModelCheckpoint(saved_model_dir+'/LSTM2-{epoch:03d}-{val_auc:03f}.h5', monitor = 'val_auc', verbose = 0, save_best_only = True,
    save_weights_only = True, mode = 'max')

csv_logger = CSVLogger(saved_model_dir+'/LSTM2-8882_training_history.csv', append=True, separator=',')
"""
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, mode = 'min')
"""
model.fit(train_dataset,
            epochs = n_epochs,
            validation_data = valid_dataset,
            callbacks = [checkpoint, csv_logger, MyCallback()],#, reduce_lr], 
            use_multiprocessing = False,
            workers = 6,
            initial_epoch = 0)
#%%
model.evaluate(test_dataset, 
                   batch_size = batch_size,
                   verbose = 1)
print('threshold =',threshold)
#%%
y_preds = np.zeros((len(merged_test), 1), dtype = 'float32')
y_trues = np.zeros((len(merged_test), 1), dtype = 'float32')
Y_preds = np.zeros((len(merged_test), 1), dtype = 'float32')
Y_trues = np.zeros((len(merged_test), 1), dtype = 'float32')
l=0
for i, ([x, y], z) in enumerate(test_dataset):
    y_true = z
    batch_size = y_true.shape[1]
    y_pred = model.predict([x, y])
    y_preds[l:l+batch_size,0] = np.squeeze(y_pred)
    y_trues[l:l+batch_size,0] = np.squeeze(y_true)
    l += batch_size
print('finished predicting')

#for i in range(1,10):
i=5
thresh = i/10
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
print('\n\n')
#%%
fpr, tpr, thresholds = roc_curve(y_trues, y_preds, pos_label=1)
np.save('roc_curve/fpr_GRU-features.npy', fpr)
np.save('roc_curve/tpr_GRU-features.npy', tpr)
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
length = 0
for i, ([x, y], z) in enumerate(test_dataset):
    length += z.shape[1]


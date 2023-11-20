#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:00:46 2021

@author: sindhura
"""
#%%
import numpy as np
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import pandas as pd
import zipfile
import pydicom
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from skimage.transform import iradon
from keras_models import getSinonet2, getSinonet1
from Datagenerator1 import CNNDataGenerator, CNNDataGeneratorEmbeds
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        grads = tf.cast(grads, "float32")
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        convOutputs = tf.cast(convOutputs, "float32")
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap            
            
    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        (h, w) = image.shape
        img = np.zeros((h, w, 3), dtype = 'uint8')
        for i in range(3):
            img[:, :, i] = image
        #image = cv2.applyColorMap(image, colormap)
        #output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        output = 0.4*heatmap + img
        output = cv2.normalize(src=output, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
    

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
models_path = 'cnn_models3'
train_images_dir = 'stage_2_train'
train = pd.read_csv('train8882.csv')
target = 'rsna-intracranial-hemorrhage-detection.zip'
theta = np.linspace(0., 180., 360)
shape = (725, 360, 1)
#model = getSinonet1(shape)
#model.load_weights(models_path+'/CNN8882-2-012-0.893494.h5')
#grad_cam = GradCAM(model, 0)
#layer_name = grad_cam.find_target_layer()
handle = zipfile.ZipFile(target)
#%%
import cv2
from PIL import Image
path1 = 'sinonet_non_windowed_sinograms/NWCT-8882'
#13, 23, 5 133
#best up till now 71, 122, 5
idx = 23#selected are 5, 71, 105, 118, 122, 123, 167, 172, 174
with handle.open(os.path.join(train_images_dir, str(train.loc[idx, 'Image'])+'.dcm')) as p:
    dicom = pydicom.dcmread(p)
    img = bsb_window(dicom)
#plt.imshow(img, cmap='gray')

image = np.load(path+'/'+train.loc[idx, 'Image']+'.npy')

image = np.squeeze(image)
u8_1_img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
u8_1_image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
final_img1 = np.zeros((512, 512, 3), dtype = 'uint8')
final_img1[:,:,0] = u8_1_img
img1 = Image.fromarray(u8_1_img)
imag1 = Image.fromarray(final_img1)
#img1.save('final_qual_images/133.png')
#imag1.save('final_qual_images/133_red.png')
np.save('final_qual_images/23.npy', u8_1_image)
#%%
fig, axs = plt.subplots(2, 4, figsize=(18, 8))

for i in range(4):
    axs[0,i].set_xticks([])
    axs[0,i].set_yticks([])
axs[0,0].imshow(u8_1_img1, cmap=plt.cm.Greys_r, aspect = 'auto')
axs[0,1].imshow(u8_1_img2, cmap=plt.cm.Greys_r, aspect = 'auto')
axs[0,2].imshow(u8_1_img3, cmap=plt.cm.Greys_r, aspect = 'auto')
am1 = axs[0,3].imshow(u8_1_img4, cmap=plt.cm.Greys_r, aspect = 'auto')

dx, dy = 0.5 * 180.0 / max(img.shape), 0.5 / image.shape[0]
#ax2.set_title("Radon transform\n(Sinogram)")
axs[1,0].set_xlabel("Projection angle (deg)", fontsize = 13)
axs[1,0].set_ylabel("Projection position (pixels)", fontsize = 13)
axs[1,0].imshow(u8_1_image1, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, image.shape[0] + dy),
           aspect='auto')
axs[1,1].imshow(u8_1_image2, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, image.shape[0] + dy),
           aspect='auto')
axs[1,2].imshow(u8_1_image3, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, image.shape[0] + dy),
           aspect='auto')
am2 = axs[1,3].imshow(u8_1_image4, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, image.shape[0] + dy),
           aspect='auto')

fig.tight_layout(pad=5)
fig.colorbar(am1, ax = axs[0,3], shrink=1)
fig.colorbar(am2, ax = axs[1,3])
#fig.text(0.5, 0.04, 'Projection angle (deg)', va='center', ha='center')
plt.show()
#%%
label = np.zeros((1, 1), dtype = 'float32')
label = np.array(train.loc[idx, 'any'], dtype = 'float32')
image = np.expand_dims(image, 0)
label = np.expand_dims(label, 0)
hmap = grad_cam.compute_heatmap(image)
u8_hmap = cv2.normalize(src=hmap, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
u8_image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)[0,:,:,0]
ct_hmap = iradon(hmap, theta=theta, filter_name='ramp', circle = False)
u8_ct_hmap =cv2.normalize(src=ct_hmap, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
u8_img =cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
heatmap, overlay_hmap = grad_cam.overlay_heatmap(u8_ct_hmap, u8_img, colormap=cv2.COLORMAP_HOT)
sino_hmap, sino_overlay_hmap = grad_cam.overlay_heatmap(u8_hmap, u8_image, colormap=cv2.COLORMAP_HOT)
plt.imshow(overlay_hmap)
#%%
cv2.imwrite('heatmaps/174_sino_img.jpg', u8_image)
cv2.imwrite('heatmaps/174_sino_hmap.jpg', sino_overlay_hmap)
cv2.imwrite('heatmaps/174_ct_img.jpg', u8_img)
cv2.imwrite('heatmaps/174_ct_hmap.jpg', overlay_hmap)




